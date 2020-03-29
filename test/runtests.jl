#!/usr/bin/env julia
using ObjectDetectionStats
using Test

@testset "Box Translate and Clamp Tests" begin
    a            = Box(5,5,15,15)
    trans_result = translate( a, -5, -5 )
    @test trans_result                          == Box(0,0,10,10)
    @test clamp(trans_result, Box(5,5,15,15))   == Box(5,5,10,10)
    @test clamp(trans_result, Box(0,0, 5, 5))   == Box(0,0, 5, 5)
end

@testset "Box Area Tests" begin
    @test area( Box(5,5,15,15) ) == 100
    @test area( Box(5,5,11,11) ) == 36
end

@testset "Box Intersection Tests" begin
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

@testset "3 TP test" begin
    ods_machine     = ObjectDetectionScore( 3 )
    pred_scores     = [ 0.2 0.9 0.5 ; #3
                        0.2 0.9 0.5;  #2
                        0.5 0.9 0.2]  #1
    pred_locations  = [ Box( 1,     1,  10,     10 ),
                        Box( 15,    1,  25,     20 ),
                        Box( 1,    15,  10,     25 ) ]
    GT_cold_encodings = [ 2, 2, 2 ]# all correct
    #prepare inputs for evaluation...
    hcl = HotClassLocalization( pred_scores, pred_locations )
    ccl = ColdClassLocalization( GT_cold_encodings, pred_locations )
    ods_machine( hcl, ccl )
    @test all(ods_machine.TP .== [0,3,0])
    @test all(ods_machine.FP .== [0,0,0])
    @test all(ods_machine.FN .== [0,0,0])
end

@testset "3 wrong class test" begin
    ods_machine     = ObjectDetectionScore( 3 )
    pred_scores     = [ 0.2 0.9 0.5 ; #3
                        0.2 0.9 0.5; #2
                        0.5 0.9 0.2]  #1
    pred_locations  = [ Box( 1,     1,  10,     10 ),
                        Box( 15,    1,  25,     20 ),
                        Box( 1,    15,  10,     25 ) ]
    GT_cold_encodings = [ 1, 1, 1 ]# all incorrect class but perfect overlap
    #prepare inputs for evaluation...
    hcl = HotClassLocalization( pred_scores, pred_locations )
    ccl = ColdClassLocalization( GT_cold_encodings, pred_locations )

    ods_machine( hcl, ccl )
    @test all(ods_machine.TP .== [0,0,0])
    @test all(ods_machine.FP .== [0,3,0])
    @test all(ods_machine.FN .== [3,0,0])
end

@testset "3 wrong class locations with correct predictions" begin
    ods_machine     = ObjectDetectionScore( 3 )
    pred_scores     = [ 0.2 0.9 0.5 ; #2
                        0.2 0.9 0.5;  #2
                        0.5 0.9 0.2 ] #2
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

    @test all(ods_machine.TP .== [0,0,0])
    @test all(ods_machine.FP .== [0,3,0])
    @test all(ods_machine.FN .== [0,3,0])
end

@testset "3 wrong class locations with wrong predictions" begin
    ods_machine     = ObjectDetectionScore( 3 )
    pred_scores     = [ 0.2 0.5 0.9; #3
                        0.2 0.9 0.5; #2
                        0.9 0.5 0.2] #1
    pred_locations  = [ Box( 1,     1,  10,     10 ),
                        Box( 15,    1,  25,     20 ),
                        Box( 1,    15,  10,     25 ) ]
    GT_cold_encodings     = [   3, #correct
                                1, #incorrect
                                1 ]#correct
    GT_locations  = [   Box( 1,     1,  10,     10 ), #TP class 3
                        Box( 15,    1,  25,     20 ), #FP class 1
                        Box( 30,    30, 42,     42 )] #FN class 1
    #prepare inputs for evaluation...
    hcl = HotClassLocalization( pred_scores, pred_locations )
    ccl = ColdClassLocalization( GT_cold_encodings, GT_locations )
    ods_machine( hcl, ccl )
    @test all(ods_machine.TP .== [0,0,1])
    @test all(ods_machine.FP .== [1,1,0])
    @test all(ods_machine.FN .== [2,0,0])
end

@testset "2 correct overlapping a prediction and 1 nonprediction" begin
    ods_machine     = ObjectDetectionScore( 3 )
    pred_scores     = [ 0.2 0.9 0.5 ; #2
                        0.2 0.9 0.5;  #2
                        0.0 0.0 0.0;] #0 not predicted
    pred_locations  = [ Box( 1,     1,  10,     10 ),
                        Box( 1,     1,  10,     10 ),
                        Box( 15,   15,  20,     20 )  ]
    GT_cold_encodings = [ 2 ]# all correct class
    GT_locations  = [   Box( 1,     1,  10,     10 ) ]
    #prepare inputs for evaluation...
    hcl = HotClassLocalization( pred_scores, pred_locations )
    ccl = ColdClassLocalization( GT_cold_encodings, GT_locations )

    ods_machine( hcl, ccl )
    @test all(ods_machine.TP .== [0,1,0])
    @test all(ods_machine.FP .== [0,1,0])
    @test all(ods_machine.FN .== [0,0,0])
end
