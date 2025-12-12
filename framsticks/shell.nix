{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  packages = with pkgs; [
    subversion
    python3
    stdenv.cc.cc.lib
    wineWowPackages.stagingFull
    parallel
    bc

    python3Packages.numpy
    python3Packages.deap
    python3Packages.matplotlib
    python3Packages.seaborn
    python3Packages.plotly
  ];

  shellHook = ''
    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"
    export WINEPREFIX=$PWD/wine
    export DIR_WITH_FRAMS_LIBRARY=$PWD/Framsticks54

    if [ ! -d "$WINEPREFIX" ]; then
      winecfg
    fi
  '';
}