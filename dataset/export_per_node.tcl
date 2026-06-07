set design_clk    $env(DESIGN)
set odb_file  $env(ODB_FILE)
set sdc_file  $env(SDC_FILE)
set tech_lef  $env(TECH_LEF)
set cell_lef  $env(CELL_LEF)
set liberty   $env(LIBERTY)

read_lef $tech_lef
read_lef $cell_lef
read_liberty $liberty
read_db $odb_file
read_sdc $sdc_file

sta::find_timing -full_update

set outdir "./dataset/pin_features_dir"
file mkdir $outdir

# pin features
set fp [open "${outdir}/pin_features_${design_clk}.csv" w]
puts $fp "pin_id,cell_type,direction,is_clk,fanout,slack_min,slew_r,slew_f"

foreach pin [get_pins -hierarchical *] {

    # reset
    set pin_id ""
    set direction ""
    set slack_min -999
    set slew_r -999
    set slew_f -999
    set is_clk -1
    set fanout 0
    set cell_type ""

    # properties
    catch {set pin_id [get_property $pin full_name]}
    catch {set direction [get_property $pin direction]}
    catch {set slack_min [get_property $pin slack_min]}
    catch {set slew_r [get_property $pin slew_max_rise]}
    catch {set slew_f [get_property $pin slew_max_fall]}
    catch {set is_clk [get_property $pin is_register_clock]}

    # fanout, default for input pins = 0
    if {$direction == "output"} {
        catch {set fanout [llength [get_fanout -from $pin]]}
    }

    # cell type
    catch {
        set cell [get_cells -of_objects $pin]
        set cell_type [get_property $cell ref_name]
    }

    puts $fp "$pin_id,$cell_type,$direction,$is_clk,$fanout,$slack_min,$slew_r,$slew_f"
}

puts "CSV generation successful: ${outdir}/pin_features_${design_clk}.csv"
close $fp
