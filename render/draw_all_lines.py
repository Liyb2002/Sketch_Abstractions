from pathlib import Path

import brep_read



current_folder = Path.cwd().parent
filename = current_folder / "output" / "root.step"
shape = brep_read.sample_strokes_from_step_file(str(filename))  # STEP reader expects a str
