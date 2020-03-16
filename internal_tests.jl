using Pkg
Pkg.activate("/home/caseykneale/.julia/dev/ObjectDetectionStats")
using ObjectDetectionStats

ods_machine = ObjectDetectionScore( 4 )

ods_machine.TP[:] .= [90,90,90,90]
ods_machine.FP[:] .= [10,10,10,10]
ods_machine.FN[:] .= [10,10,10,10]

classwise_precision( ods_machine )
classwise_recall( ods_machine )
macro_precision(ods_machine)
macro_recall(ods_machine)
