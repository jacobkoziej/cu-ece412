{
  description = "The Cooper Union - ECE 412: Speech and Audio Signal Processing";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
  };

  outputs =
    inputs:

    inputs.flake-utils.lib.eachDefaultSystem (
      system:

      let
        pkgs = import inputs.nixpkgs {
          inherit system;

          config = {
            allowUnfree = true;
          };
        };

        python = pkgs.python3;

        python-pkgs =
          (python.withPackages (
            python-pkgs: with python-pkgs; [
              ipython
              einops
              jupyter
              jupytext
              openai-whisper
              papermill
              pytorch-lightning
              tensorboard
              torchWithCuda
              torchaudio
              evaluate
              jiwer
            ]
          )).override
            (args: {
              ignoreCollisions = true;
            });

      in
      {
        devShells.default = pkgs.mkShell {
          packages =
            [
              python-pkgs
            ]
            ++ (with pkgs; [
              black
              scons
              treefmt2
            ]);
        };

        formatter = pkgs.nixfmt-rfc-style;
      }
    );
}
