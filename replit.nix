{ pkgs }: {
  deps = [
    pkgs.python311Full
    pkgs.replitPackages.prybar-python311
    pkgs.replitPackages.stderred
    pkgs.gcc
    pkgs.pkg-config
    pkgs.cairo
    pkgs.pango
    pkgs.gdk-pixbuf
    pkgs.gtk3
    pkgs.gobject-introspection
    pkgs.libffi
  ];
}