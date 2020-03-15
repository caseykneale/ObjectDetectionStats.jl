using Pkg
Pkg.activate("ObjectDetectionStats/")
using ObjectDetectionStats

ods_machine     = ObjectDetectionScore( 3 )
pred_scores     = [ 0.2 0.9 0.5 ; #3
                    0.2 0.9 0.5; #2
                    0.5 0.9 0.2  #1
                  ]
pred_locations  = [ Box( 1,     1,  10,     10 ),
                    Box( 15,    1,  25,     20 ),
                    Box( 1,    15,  10,     25 ) ]

GT_cold_encodings = [ 2, 2, 2 ]# all correct class

GT_locations  = [   Box( 100,    100,  100,     100 ),
                    Box( 150,    100,  250,     200 ),
                    Box( 100,    150,  100,     250 ) ]

#prepare inputs for evaluation...
hcl = HotClassLocalization( pred_scores, pred_locations )
ccl = ColdClassLocalization( GT_cold_encodings, GT_locations )

ods_machine( hcl, ccl )
ods_machine

@test all(ods_machine.TP .== [0,0,0])
@test all(ods_machine.FP .== [0,3,0])
@test all(ods_machine.FN .== [0,0,0])
