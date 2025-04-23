{
  description = "BEAST Python dev environment with Conda (micromamba)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs = { self, nixpkgs }: {
    devShells = {
      default = let
        pkgs = import nixpkgs { system = "aarch64-darwin"; };  # Specify the right system type here
        micromamba-shell = pkgs.callPackage (pkgs.fetchFromGitHub {
          owner = "micromamba";
          repo = "micromamba-nix";
          rev = "v0.2.12";
          sha256 = "1qkvd9cpdfj2mb7xjmkg74iw8g9y7zha0gfp0wjyz0iyxzcrhv9i";  # Replace with correct hash
        }) {};
      in micromamba-shell.override {
        environmentFile = ./environment.yml;
      };
    };
  };
}
