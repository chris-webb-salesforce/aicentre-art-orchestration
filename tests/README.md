# Test Files

Place test assets in the `fixtures/` directory.

## Structure

```
tests/
├── fixtures/           # Test input files (committed to git)
│   ├── sample_portrait.jpg
│   ├── sample_lineart.png
│   └── ...
└── README.md
```

## Usage

Reference fixtures in tests like:

```bash
python scripts/test_components.py contours --image tests/fixtures/sample_lineart.png
python scripts/test_components.py openai --image tests/fixtures/sample_portrait.jpg
python scripts/test_components.py gcode --image tests/fixtures/sample_lineart.png
```

## Notes

- Keep fixture files small (under 1MB) for git
- Generated output goes to `output/` (gitignored)
- Temporary test files in root (test_*.jpg, test_*.png) are gitignored
