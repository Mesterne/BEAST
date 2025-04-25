{
  description = "BEAST Python dev environment with Conda (miniconda)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }: 
  let
    supportedSystems = [ "x86_64-darwin" "aarch64-darwin" "x86_64-linux" "aarch64-linux" ];
    forAllSystems = f: builtins.listToAttrs (map (system: { name = system; value = f system; }) supportedSystems);
  in {
    devShells = forAllSystems (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in {
        default = pkgs.mkShell {
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

            exec zsh -l
          '';
        };
      }
    );
  };
}
