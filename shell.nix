# To enable this nix shell, use nix shell
let
  pkgs = import <nixpkgs> {};

  
in pkgs.mkShell {
  packages = with pkgs; [
    php
  ];
}
