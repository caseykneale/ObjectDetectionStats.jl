using Pkg
Pkg.activate("ObjectDetectionStats/")
using ObjectDetectionStats

Classes = 3
ods_machine = ObjectDetectionScore( 3 )

#So...  we need to


pred_scores     = [ 0.2 0.5 0.9; #3
                    0.2 0.9 0.5; #2
                    0.9 0.5 0.2  #1
                  ]
pred_locations  = [ Box( 1,     1,  10,     10 ),
                    Box( 15,    1,  25,     20 ),
                    Box( 1,    15,  10,     25 ),
                  ]

GT_cold_encodings     = [   3, #correct
                            1, #incorrect
                            1 ]#correct
GT_locations  = [   Box( 1,     1,  10,     10 ), #TP class 3
                    Box( 15,    1,  25,     20 ), #FP class 1
                    Box( 30,    30, 42,     42 ), #FN class 1
                  ]

#prepare inputs for evaluation...
hcl = HotClassLocalization( pred_scores, pred_locations )
ccl = ColdClassLocalization( GT_cold_encodings, GT_locations )

ods_machine
( ods_machine )( hcl, ccl )
ods_machine
