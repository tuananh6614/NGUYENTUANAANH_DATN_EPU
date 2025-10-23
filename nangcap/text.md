Traceback (most recent call last):
  File "E:\FIRMWAVE\project\run.py", line 99, in <module>
    time.sleep(1)
KeyboardInterrupt

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "E:\FIRMWAVE\project\run.py", line 103, in <module>
    stop_app()
  File "E:\FIRMWAVE\project\run.py", line 71, in stop_app
    p.wait(timeout=5)  # ✅ FIX: Tăng timeout
    ^^^^^^^^^^^^^^^^^
  File "C:\Users\Admin\AppData\Local\Programs\Python\Python311\Lib\subprocess.py", line 1264, in wait
    return self._wait(timeout=timeout)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Admin\AppData\Local\Programs\Python\Python311\Lib\subprocess.py", line 1590, in _wait
    result = _winapi.WaitForSingleObject(self._handle