from __future__ import annotations

import ctypes
import datetime
import os
import sys
from dataclasses import dataclass
from typing import Iterator, Optional


class EverythingSdkError(RuntimeError):
    pass


# Everything request flags (subset)
EVERYTHING_REQUEST_FILE_NAME = 0x00000001
EVERYTHING_REQUEST_PATH = 0x00000002
EVERYTHING_REQUEST_FULL_PATH_AND_FILE_NAME = 0x00000004
EVERYTHING_REQUEST_SIZE = 0x00000010
EVERYTHING_REQUEST_DATE_MODIFIED = 0x00000040
EVERYTHING_REQUEST_ATTRIBUTES = 0x00000100


@dataclass(frozen=True)
class EverythingResult:
    full_path: str
    file_name: str
    path: str
    size: Optional[int]
    date_modified_utc: Optional[datetime.datetime]
    is_folder: Optional[bool] = None


def _is_windows() -> bool:
    return sys.platform == "win32"


def _default_dll_name() -> str:
    # Prefer 64-bit if Python is 64-bit.
    is_64 = ctypes.sizeof(ctypes.c_void_p) == 8
    return "Everything64.dll" if is_64 else "Everything32.dll"


def resolve_dll_path(explicit: str | None, *, base_dir: str) -> str:
    if explicit:
        p = os.path.abspath(explicit)
        if not os.path.exists(p):
            raise EverythingSdkError(f"Everything DLL not found: {p}")
        return p

    env = os.environ.get("EVERYTHING_SDK_DLL")
    if env and env.strip():
        p = os.path.abspath(env.strip())
        if not os.path.exists(p):
            raise EverythingSdkError(f"EVERYTHING_SDK_DLL points to missing file: {p}")
        return p

    local = os.path.join(base_dir, _default_dll_name())
    if os.path.exists(local):
        return os.path.abspath(local)

    raise EverythingSdkError(
        "Everything SDK DLL not found. Place Everything64.dll next to this script or set EVERYTHING_SDK_DLL."
    )


# FILETIME struct used by Everything_GetResultDateModified
class _FILETIME(ctypes.Structure):
    _fields_ = [("dwLowDateTime", ctypes.c_uint32), ("dwHighDateTime", ctypes.c_uint32)]


def _filetime_to_dt_utc(ft: _FILETIME):
    import datetime as dt

    # 100-nanosecond intervals since 1601-01-01 UTC
    v = (int(ft.dwHighDateTime) << 32) + int(ft.dwLowDateTime)
    if v <= 0:
        return None
    return dt.datetime(1601, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(microseconds=v / 10)


class EverythingClient:
    def __init__(self, dll_path: str):
        if not _is_windows():
            raise EverythingSdkError("Everything SDK is Windows-only (win32).")

        try:
            self._dll = ctypes.WinDLL(dll_path)
        except Exception as e:
            raise EverythingSdkError(f"Failed to load Everything DLL: {dll_path}: {e}")

        self._bind()

    def _bind(self) -> None:
        d = self._dll

        # void Everything_SetSearchW(LPCWSTR)
        d.Everything_SetSearchW.argtypes = [ctypes.c_wchar_p]
        d.Everything_SetSearchW.restype = None

        # void Everything_SetRequestFlags(DWORD)
        d.Everything_SetRequestFlags.argtypes = [ctypes.c_uint32]
        d.Everything_SetRequestFlags.restype = None

        # void Everything_SetSort(DWORD)
        d.Everything_SetSort.argtypes = [ctypes.c_uint32]
        d.Everything_SetSort.restype = None

        # BOOL Everything_QueryW(BOOL wait)
        d.Everything_QueryW.argtypes = [ctypes.c_bool]
        d.Everything_QueryW.restype = ctypes.c_bool

        # DWORD Everything_GetNumResults(void)
        d.Everything_GetNumResults.argtypes = []
        d.Everything_GetNumResults.restype = ctypes.c_uint32

        # LPCWSTR Everything_GetResultFileNameW(DWORD index)
        d.Everything_GetResultFileNameW.argtypes = [ctypes.c_uint32]
        d.Everything_GetResultFileNameW.restype = ctypes.c_wchar_p

        # void Everything_GetResultFullPathNameW(DWORD index, LPWSTR buf, DWORD bufSize)
        d.Everything_GetResultFullPathNameW.argtypes = [ctypes.c_uint32, ctypes.c_wchar_p, ctypes.c_uint32]
        d.Everything_GetResultFullPathNameW.restype = None

        # BOOL Everything_GetResultSize(DWORD index, LARGE_INTEGER* size)
        d.Everything_GetResultSize.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_int64)]
        d.Everything_GetResultSize.restype = ctypes.c_bool

        # BOOL Everything_GetResultDateModified(DWORD index, FILETIME* ft)
        d.Everything_GetResultDateModified.argtypes = [ctypes.c_uint32, ctypes.POINTER(_FILETIME)]
        d.Everything_GetResultDateModified.restype = ctypes.c_bool

        # DWORD Everything_GetResultAttributes(DWORD index)
        try:
            d.Everything_GetResultAttributes.argtypes = [ctypes.c_uint32]
            d.Everything_GetResultAttributes.restype = ctypes.c_uint32
        except Exception:
            pass

        # Optional match settings (exist in Everything SDK)
        for name, argtypes in (
            ("Everything_SetMatchCase", [ctypes.c_bool]),
            ("Everything_SetMatchWholeWord", [ctypes.c_bool]),
            ("Everything_SetRegex", [ctypes.c_bool]),
        ):
            try:
                fn = getattr(d, name)
                fn.argtypes = argtypes
                fn.restype = None
            except Exception:
                # Older SDKs should still function without these.
                pass

        # DWORD Everything_GetLastError(void)
        try:
            d.Everything_GetLastError.argtypes = []
            d.Everything_GetLastError.restype = ctypes.c_uint32
        except Exception:
            pass

    def query(
        self,
        *,
        search: str,
        limit: int = 50,
        match_case: bool = False,
        whole_word: bool = False,
        regex: bool = False,
        request_size: bool = True,
        request_date_modified: bool = True,
        request_attributes: bool = True,
    ) -> Iterator[EverythingResult]:
        search = (search or "").strip()
        if not search:
            return iter(())

        d = self._dll

        # Set match behavior (best-effort)
        for name, val in (
            ("Everything_SetMatchCase", match_case),
            ("Everything_SetMatchWholeWord", whole_word),
            ("Everything_SetRegex", regex),
        ):
            try:
                getattr(d, name)(bool(val))
            except Exception:
                pass

        flags = EVERYTHING_REQUEST_FULL_PATH_AND_FILE_NAME | EVERYTHING_REQUEST_FILE_NAME | EVERYTHING_REQUEST_PATH
        if request_size:
            flags |= EVERYTHING_REQUEST_SIZE
        if request_date_modified:
            flags |= EVERYTHING_REQUEST_DATE_MODIFIED
        if request_attributes:
            flags |= EVERYTHING_REQUEST_ATTRIBUTES

        d.Everything_SetSearchW(search)
        d.Everything_SetRequestFlags(flags)

        ok = bool(d.Everything_QueryW(True))
        if not ok:
            err = None
            try:
                err = int(d.Everything_GetLastError())
            except Exception:
                err = None
            raise EverythingSdkError(f"Everything_QueryW failed" + (f" (err={err})" if err is not None else ""))

        n = int(d.Everything_GetNumResults())
        if n <= 0:
            return iter(())

        lim = max(1, min(10000, int(limit)))
        count = min(n, lim)

        # Buffer for full path.
        buf_len = 4096
        buf = ctypes.create_unicode_buffer(buf_len)

        def _iter() -> Iterator[EverythingResult]:
            for i in range(count):
                # filename
                fn = d.Everything_GetResultFileNameW(i) or ""

                # full path
                buf.value = ""
                d.Everything_GetResultFullPathNameW(i, buf, buf_len)
                full = buf.value or ""

                # path
                path = ""
                if full:
                    path = os.path.dirname(full)

                # size
                size: Optional[int] = None
                if request_size:
                    out = ctypes.c_int64(0)
                    try:
                        if bool(d.Everything_GetResultSize(i, ctypes.byref(out))):
                            size = int(out.value)
                    except Exception:
                        size = None

                # date modified
                dt_utc = None
                if request_date_modified:
                    ft = _FILETIME()
                    try:
                        if bool(d.Everything_GetResultDateModified(i, ctypes.byref(ft))):
                            dt_utc = _filetime_to_dt_utc(ft)
                    except Exception:
                        dt_utc = None

                # attributes (folder/file)
                is_folder = None
                if request_attributes:
                    try:
                        attrs = int(d.Everything_GetResultAttributes(i))
                        # FILE_ATTRIBUTE_DIRECTORY = 0x10
                        if attrs != 0xFFFFFFFF:
                            is_folder = bool(attrs & 0x10)
                    except Exception:
                        is_folder = None

                yield EverythingResult(
                    full_path=full,
                    file_name=str(fn),
                    path=path,
                    size=size,
                    date_modified_utc=dt_utc,
                    is_folder=is_folder,
                )

        return _iter()
