# verl Documentation

We recommend new contributors start from writing documentation, which helps you quickly understand the verl codebase. Most documentation files are located under the `docs/` folder.

## Build the docs

```bash
# Install dependencies
pip install -r requirements-docs.txt

# Build the docs
make clean
make html
```

## Open the docs with your browser

```bash
python -m http.server -d _build/html/
```
Launch your browser and open localhost:8000.

## Documentation Workflow

### Update Documentation

Update your documentation files in the appropriate subdirectories under `docs/`. If you add new files, remember to update `index.rst` (or relevant `.rst` files) accordingly.

- **`pre-commit run --all-files`** manually runs all configured checks, applying fixes if possible. If it fails the first time, re-run it to ensure lint errors are fully resolved. Make sure your code passes all checks **before** creating a Pull Request.
- **Do not commit** directly to the `main` branch. Always create a new branch (e.g., `feature/my-new-feature`), push your changes, and open a PR from that branch.

```bash
# 1) Compile all Jupyter notebooks (if any)
make compile

# 2) Compile and Preview documentation locally with auto-build
# This will automatically rebuild docs when files change
# Open your browser at the displayed port to view the docs
make serve

# 2a) Alternative ways to serve documentation
# With custom port
PORT=8080 make serve

# 3) Clean notebook outputs (if any)
# nbstripout removes notebook outputs so your PR stays clean
pip install nbstripout
find . -name '*.ipynb' -exec nbstripout {} \;

# 4) Pre-commit checks and create a PR
# After these checks pass, push your changes and open a PR on your branch
pre-commit run --all-files
```

### Documentation Best Practices

1. **Keep it Simple**: Write clear, concise documentation that is easy to understand.
2. **Include Examples**: Provide practical examples for each feature or concept.
3. **Stay Up-to-Date**: Keep documentation in sync with code changes.
4. **Use Proper Formatting**: Follow RST and Markdown formatting guidelines.
5. **Cross-Reference**: Use proper cross-references between different sections.
6. **Test Examples**: If including code examples, ensure they are tested and working.

### Port Allocation

When serving documentation locally, you can specify a custom port to avoid conflicts:

```bash
PORT=8080 make serve
```

This will start the documentation server on port 8080 instead of the default port.
