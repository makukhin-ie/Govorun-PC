' Тихий запуск Govorun PC: без окна терминала, но с логом.
'
' Оригинальная версия звала pythonw и выбрасывала весь вывод. При тихом
' старте это означало, что при любой поломке — не тот микрофон, не
' скачалась модель, занят порт второй копией — программа просто молча
' не поднималась, и выяснить причину было негде.
'
' Здесь консоль тоже скрыта, но вывод идёт в govorun.log рядом со
' скриптом. Файл перезаписывается при каждом запуске, так что в нём
' всегда последняя попытка.

Option Explicit

Dim fso, sh, folder, command

Set fso = CreateObject("Scripting.FileSystemObject")
Set sh  = CreateObject("WScript.Shell")

folder = fso.GetParentFolderName(WScript.ScriptFullName)

command = "cmd /c cd /d """ & folder & """ && " & _
          "python govorun_pc.py > govorun.log 2>&1"

' 0 — окно скрыто, False — не ждать завершения
sh.Run command, 0, False