import argparse
import subprocess
import os
import gzip
import shutil
import tempfile
import re

def prepare_odb(path):
    if not path.endswith(".gz"):
        return path

    tmp = tempfile.NamedTemporaryFile(
        suffix=".odb",
        delete=False
    )

    with gzip.open(path, "rb") as fin:
        shutil.copyfileobj(fin, tmp)

    tmp.close()
    return tmp.name


def run(design, odb, sdc, tcl, tech_lef, cell_lef, liberty):

    tcl = os.path.abspath(tcl)
    odb = os.path.abspath(odb)
    sdc = os.path.abspath(sdc)
    tech_lef = os.path.abspath(tech_lef)
    cell_lef = os.path.abspath(cell_lef)
    liberty = os.path.abspath(liberty)

    env = os.environ.copy()
    env["DESIGN"] = design
    env["ODB_FILE"] = odb
    env["SDC_FILE"] = sdc
    env["TECH_LEF"] = tech_lef
    env["CELL_LEF"] = cell_lef
    env["LIBERTY"] = liberty

    subprocess.run(
        ["openroad", "-exit", tcl],
        check=True,
        env=env
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--build_dir", required=True,
                        help="Path to build directory")

    parser.add_argument("--tcl", required=True,
                        help="Path to export_per_node TCL script")

    parser.add_argument("--tech_lef", required=True,
                        help="Technology LEF (.tlef)")

    parser.add_argument("--cell_lef", required=True,
                        help="Standard cell LEF")

    parser.add_argument("--liberty", required=True,   # synthesis, inputs, .lib
                        help="Liberty timing library (.lib), usually in build/.../synthesis/inputs/.lib")

    args = parser.parse_args()

    subprocess.run(["sed", "-i", "s/\\r//", args.tcl], check=True)

    for design_clk in os.listdir(args.build_dir):

        # set paths
        design = design_clk.rsplit("_", 1)[0]

        odb_path_raw = os.path.join(args.build_dir, f"{design_clk}/job0/write.views/0/outputs/{design}.odb.gz")
        sdc_path = os.path.join(args.build_dir, f"{design_clk}/job0/write.views/0/inputs/{design}.sdc")

        # invalid paths
        if not os.path.exists(odb_path_raw) or not os.path.exists(sdc_path):
            print(f"Skipping {design}, missing files")
            continue

        # extract .gz
        odb_path = prepare_odb(odb_path_raw)

        try:
            run(
                design_clk,
                odb_path,
                sdc_path,
                args.tcl,
                args.tech_lef,
                args.cell_lef,
                args.liberty
            )

        finally:
            if odb_path != odb_path_raw:
                os.remove(odb_path)





