; Auto-Sorter Inno Setup Installer Script
; Produces a standalone, single-file Windows installer that operates 100% offline.

#define MyAppName "Auto-Sorter"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "AutoSorter"
#define MyAppURL "https://github.com/artic2702/auto-sorter"
#define MyAppExeName "AutoSorter.exe"

[Setup]
; Basic Application Info
AppId={{E61A0512-C82D-4D39-847F-5A2C873D963F}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}

; Installation directories and privileges
; "lowest" allows standard user installs without UAC prompt;
; "dialog" lets the user select between Current User and All Users (Admin).
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog commandline
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes

; Visuals & Branding
SetupIconFile=app.ico
UninstallDisplayIcon={app}\{#MyAppExeName}

; Output configuration
OutputDir=..\dist\installer
OutputBaseFilename=AutoSorter-Setup-{#MyAppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern

; Architectures
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Copy all files produced by PyInstaller onedir build
Source: "..\dist\AutoSorter\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
