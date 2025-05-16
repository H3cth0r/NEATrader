{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.python3
    pkgs.python3Packages.numpy
    pkgs.python3Packages.pandas
    pkgs.python3Packages.virtualenv
  ];

  shellHook = ''
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
      python -m venv venv --system-site-packages
    fi
    
    # Activate the virtual environment
    source venv/bin/activate
    
    # Install neat-python
    pip install --quiet neat-python
  '';
}
