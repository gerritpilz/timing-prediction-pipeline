from multiprocessing import freeze_support
from siliconcompiler import ASIC, Design
from siliconcompiler.targets import skywater130_demo
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--rtl', nargs="+", required=True)
    parser.add_argument('--sdc', required=True)
    parser.add_argument('--clk_period', required=True)
    parser.add_argument('--design', required=True)   # name
    parser.add_argument('--top_module', required=True) # name in rtl topmodule

    args = parser.parse_args()

    freeze_support()

    design = Design(f"{args.design}_{args.clk_period}")
    design.set_topmodule(args.top_module, fileset="rtl")

    for f in args.rtl:
        design.add_file(f, fileset="rtl")

    design.add_file(args.sdc, fileset="sdc")

    project = ASIC(design)
    project.add_fileset(["rtl", "sdc"])
    project.option.set_remote(False)

    skywater130_demo(project)

    project.run()




















'''
from multiprocessing import freeze_support
from siliconcompiler import ASIC, Design
from siliconcompiler.targets import skywater130_demo
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--verilog', required=True)
    parser.add_argument('--cell_pin_direction', required=True)
    parser.add_argument('--cell_to_idx', required=True)
    parser.add_argument('--pin_features', required=True)
    args = parser.parse_args()
    freeze_support()

    design = Design("gcd")
    design.set_topmodule("gcd", fileset="rtl")
    design.add_file("gcd.v",   fileset="rtl")
    design.add_file("gcd.sdc", fileset="sdc")

    project = ASIC(design)
    project.add_fileset(["rtl", "sdc"])
    project.option.set_remote(False)

    skywater130_demo(project)

    project.option.add_to("place.detailed", clobber=True)
    #project.add_postscript("place.detailed", "export_slack.tcl")
    project.run()
'''