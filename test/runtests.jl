#!/usr/bin/env julia
using ObjectDetectionStats
using Test

@testset "Areas" begin
    @test area( Box(5,5,15,15) ) == 100
    @test area( Box(5,5,11,11) ) == 36
end

@testset "Intersections" begin
    a = Box(5,5,15,15)
    b = Box(5,5,15,15)
    @test intersection_area( a, b ) == 100
    @test intersection_over_union( a, b ) == 1.0

    a = Box(10,10,15,15)
    b = Box(5,5,15,15)
    @test intersection_area( a, b ) == 25
    @test intersection_over_union( a, b ) == 0.25

    b = Box(10,10,15,15)
    a = Box(5,5,15,15)
    @test intersection_area( a, b ) == 25
    @test intersection_over_union( a, b ) == 0.25

    a = Box(1,1,15,15)
    b = Box(25,25,35,35)
    @test intersection_area( a, b ) == 0
    @test intersection_over_union( a, b ) == 0.0
end

@testset "Statistics"
    ods_machine = ObjectDetectionScore( 3 )
    pred_scores     = [ 0.2 0.5 0.9; #3
                        0.2 0.9 0.5; #2
                        0.9 0.5 0.2  #1
                      ]
    pred_locations  = [ Box( 1,     1,  10,     10 ),
                        Box( 15,    1,  25,     20 ),
                        Box( 1,    15,  10,     25 ) ]

    GT_cold_encodings     = [   3, #correct
                                1, #incorrect
                                1 ]#correct
    GT_locations  = [   Box( 1,     1,  10,     10 ), #TP class 3
                        Box( 15,    1,  25,     20 ), #FP class 1
                        Box( 30,    30, 42,     42 ) #FN class 1
                      ]
    #prepare inputs for evaluation...
    hcl = HotClassLocalization( pred_scores, pred_locations )
    ccl = ColdClassLocalization( GT_cold_encodings, GT_locations )

    ods_machine( hcl, ccl )
    @test ods_machine.TP == [0,0,1]
    @test ods_machine.FP == [1,0,0]
    @test ods_machine.FN == [1,0,0]
end
