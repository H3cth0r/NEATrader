{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.python311  # Use Python 3.11 (instead of non-existent 3.13)
    pkgs.python311Packages.numpy
    pkgs.python311Packages.pandas
    pkgs.python311Packages.virtualenv
    pkgs.libffi     # Required for cffi compilation
  ];

  # Set environment variables for CFFI compilation
  NIX_CFLAGS_COMPILE = "-I${pkgs.libffi.dev}/include";
  NIX_LDFLAGS = "-L${pkgs.libffi}/lib";

  shellHook = ''
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
      python -m venv venv
    fi
    
    # Activate the virtual environment
    source venv/bin/activate
    
    # Install required packages
    pip install --upgrade pip wheel
    pip install neat-python cffi yfinance scikit-learn plotly ta torch 
  '';
}
