using Pkg
Pkg.activate("/home/caseykneale/.julia/dev/ObjectDetectionStats")
using ObjectDetectionStats

ods_machine     = ObjectDetectionScore( 3 )
pred_scores     = [ 0.2 0.9 0.5 ; #3
                    0.2 0.9 0.5; #2
                    0.5 0.9 0.2  #1
                  ]
pred_locations  = [ Box( 1,     1,  10,     10 ),
                    Box( 15,    1,  25,     20 ),
                    Box( 1,    15,  10,     25 ) ]
GT_cold_encodings = [ 1, 1, 1 ]# all incorrect class but perfect overlap
#prepare inputs for evaluation...
hcl = HotClassLocalization( pred_scores, pred_locations )
ccl = ColdClassLocalization( GT_cold_encodings, pred_locations )
ods_machine( hcl, ccl )
ods_machine



@test all(ods_machine.TP .== [0,0,0])
@test all(ods_machine.FP .== [0,0,0])
@test all(ods_machine.FN .== [0,3,0])
