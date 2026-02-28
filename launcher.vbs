' WhisperX Subtitle Generator v2.5 - Launcher
' Uruchamia aplikacje w wirtualnym srodowisku (.venv) bez okna konsoli

Option Explicit

Dim fso
Dim strDir
Dim strScript
Dim strPythonW
Dim strCommand
Dim objShell

Set fso = CreateObject("Scripting.FileSystemObject")
Set objShell = CreateObject("WScript.Shell")

' Katalog w ktorym lezy ten plik .vbs
strDir = fso.GetParentFolderName(WScript.ScriptFullName)

' Sciezki
strScript  = strDir & "\subtitle_generator.py"
strPythonW = strDir & "\.venv\Scripts\pythonw.exe"

' --- Sprawdz czy venv istnieje ---
If Not fso.FileExists(strPythonW) Then
    MsgBox "Virtual environment not found!" & vbCrLf & vbCrLf & _
           "Expected: " & strPythonW & vbCrLf & vbCrLf & _
           "Please run install.bat first.", _
           vbCritical, "WhisperX Subtitle Generator"
    WScript.Quit 1
End If

' --- Sprawdz czy skrypt istnieje ---
If Not fso.FileExists(strScript) Then
    MsgBox "Main script not found!" & vbCrLf & vbCrLf & _
           "Expected: " & strScript & vbCrLf & vbCrLf & _
           "Make sure launcher.vbs is in the same folder as subtitle_generator.py.", _
           vbCritical, "WhisperX Subtitle Generator"
    WScript.Quit 1
End If

' --- Uruchom pythonw.exe z venv (brak okna konsoli) ---
strCommand = """" & strPythonW & """ """ & strScript & """"

objShell.CurrentDirectory = strDir

On Error Resume Next
objShell.Run strCommand, 0, False   ' 0 = ukryte okno, False = nie czekaj

If Err.Number <> 0 Then
    MsgBox "Failed to launch the application." & vbCrLf & vbCrLf & _
           "Command: " & strCommand & vbCrLf & _
           "Error: " & Err.Description, _
           vbCritical, "WhisperX Subtitle Generator"
End If

Set fso = Nothing
Set objShell = Nothing
