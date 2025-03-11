{
  description = "Nix develop shell with Conda, zsh, and Neovim Python integration";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.simpleFlake {
      inherit self nixpkgs;
      devShells.default = { pkgs }:
        pkgs.mkShell {
          nativeBuildInputs = [
            pkgs.zsh
            pkgs.python311  # Ensure the correct Python version
            pkgs.neovim
            pkgs.micromamba  # Lightweight Conda alternative
          ];

          shellHook = ''
            export SHELL=$(which zsh)
            exec zsh  # Ensure the shell starts as zsh

            # Set up Conda environment
            export MAMBA_ROOT_PREFIX=$HOME/.mamba
            micromamba shell init -s zsh
            micromamba env create -f environment.yml

            # Ensure Neovim uses the correct Python
            export PYTHONPATH=$(which python)
            echo "Neovim will use Python at $PYTHONPATH"
          '';
        };
    };
}
