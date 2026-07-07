# vhs tape for the README demo GIF.
# Regenerate with:  cd docs/demo && vhs demo.tape

Output ../demo.gif

Set FontSize 20
Set Width 1200
Set Height 600
Set Padding 24
Set Theme "Catppuccin Mocha"
Set TypingSpeed 35ms

Hide
Type `export PATH="$PWD/../../.venv/bin:$PATH" && clear`
Enter
Show

Type "# 1. RECORD -- every model + tool call crosses one Effect boundary"
Enter
Sleep 600ms
Type "python record.py"
Enter
Sleep 3.5s

Type "# 2. READ the trace"
Enter
Sleep 600ms
Type "loom timeline trace.json"
Enter
Sleep 4s

Hide
Type "clear"
Enter
Show

Type "# 3. REPLAY -- offline, zero API calls, identical output"
Enter
Sleep 600ms
Type "loom replay trace.json"
Enter
Sleep 3.5s

Type "# 4. REWIND -- fork at turn 0, edit the question, take the other branch"
Enter
Sleep 600ms
Type "python rewind.py"
Enter
Sleep 6s
