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

        lib = pkgs.lib;

        python = pkgs.python3;

        python-pkgs =
          (python.withPackages (
            python-pkgs: with python-pkgs; [
              einops
              ipython
              jiwer
              jupyter
              jupytext
              librosa
              matplotlib
              numpy
              openai-whisper
              pandas
              papermill
              pytorch-lightning
              scipy
              tensorboard
              torch-audiomentations
              torchWithCuda
              torchaudio
            ]
          )).override
            (args: {
              ignoreCollisions = true;
            });

      in
      {
        devShells.default = pkgs.mkShell (
          let
            pre-commit-bin = "${lib.getBin pkgs.pre-commit}/bin/pre-commit";
          in
          {
            packages =
              [
                python-pkgs
              ]
              ++ (with pkgs; [
                black
                mdformat
                pre-commit
                ruff
                scons
                shfmt
                toml-sort
                treefmt2
                yamlfmt
              ]);

            shellHook = ''
              ${pre-commit-bin} install --allow-missing-config > /dev/null
            '';
          }
        );

        formatter = pkgs.nixfmt-rfc-style;
      }
    );
}
