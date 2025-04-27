{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    black
    isort
    micromamba
    zsh
  ];
  shellHook = ''
    export SHELL=$(which zsh)
    export MAMBA_ROOT_PREFIX="$PWD/.mamba"
    export PATH="$MAMBA_ROOT_PREFIX/bin:$PATH"

    # Initialize micromamba for zsh
    eval "$(micromamba shell hook --shell zsh)"

    if [ ! -d "$MAMBA_ROOT_PREFIX/envs/BEAST_ENV" ]; then
      echo "Creating micromamba environment from environment.yml..."
      micromamba create -n BEAST_ENV -f environment.yml -y
    fi

    micromamba activate BEAST_ENV
  '';
}
