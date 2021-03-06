using Pkg
Pkg.activate(".")
using DataFrames

global ULLR_required_columns = [ :upper_left_x, :lower_right_x, :upper_left_y, :lower_right_y ]

"""
    has_required_columns( df::DataFrame, required_columns::Vector )::Bool

Returns `true` if a `DataFrame` instance has all of the `required_columns` present.

"""
function has_required_columns( df::DataFrame, required_columns::Vector )::Bool
    df_names = names( df )
    result = true
    for required_column in required_columns
        if !( required_column in df_names )
            result = false
            break
        end
    end
    return result
end

"""
    highest_vote!( df::DataFrame, class_columns::Vector )::DataFrame

Finds highest vote amongst all specified class columns and creates a new column with that variable name.

"""
function highest_vote!( df::DataFrame, class_columns::Vector )::DataFrame
    df_classes = df[ !, class_columns ]
    df[ !, :Highest_Vote ] = map( row -> argmax( row ), eachrow( df_classes ) )
    df_classes = df[ !, vcat( class_columns, :Highest_Vote ) ]
    df[ !, :Highest_Prediction ] = map( row -> row[row.Highest_Vote], eachrow( df_classes ) )
    return df
end

highest_vote!( dfs::Vector{DataFrame}, class_columns::Vector )::Vector{DataFrame} = [ highest_vote!(df, class_columns) for df in dfs ]

"""
    barycenter_boxes!( df::DataFrame )::DataFrame

Will convert a DataFrame containing the following variables: center_x, center_y, width, height to be formatted in upper left lower right coordinates.

"""
function barycenter_boxes!( df::DataFrame )::DataFrame
    req_cols = [ :center_x, :center_y, :width, :height ]
    @assert( has_required_columns( df, req_cols ), "Input dataframe does not have the required columns: $req_cols. Please check spelling and capitalization of column names." )
    df[ !, [ :width, :height] ] ./= 2
    df[ !, :upper_left_x ] = df.center_x - df.width
    df[ !, :lower_right_x] = df.center_x + df.width
    df[ !, :upper_left_y ] = df.center_y - df.height
    df[ !, :lower_right_y] = df.center_y + df.height
    select!( df, Not( req_cols ) )
    return df
end

"""
    LLUR_boxes!( df::DataFrame )::DataFrame
"""
function LLUR_boxes!( df::DataFrame )::DataFrame
    req_cols = [ :lower_left_x, :upper_right_x, :lower_left_y, :upper_right_y ]
    new_cols = ULLR_required_columns
    @assert( has_required_columns( df, req_cols ), "Input dataframe does not have the required columns: $req_cols. Please check spelling and capitalization of column names." )
    rename!( df, Dict( req_cols .=> new_cols ) )
    return df
end

"""
    area( df::DataFrame )::DataFrame

Calculates the areas of all Boxes in a DataFrame as a new column.

"""
function area!( df::DataFrame )::DataFrame
    @assert( has_required_columns( df, ULLR_required_columns ), "Input dataframe does not have the required columns: $ULLR_required_columns. Please check spelling and capitalization of column names and ensure the format is ULLR." )
    df[ !, :Area ] = ( df.lower_right_x - df.upper_left_x ) .* ( df.lower_right_y - df.upper_left_y )
    return df
end

area!( dfs::Vector{DataFrame} )::Vector{DataFrame} = [ area!(df) for df in dfs ]

"""
    intersection_area( a::DataFrame, b::DataFrame )::DataFrame

Calculates the area of intersection between 2 DataFrames containing ULLR boxes.
This returns a new DataFrame

"""
function intersection_over_union( a::DataFrame, b::DataFrame )::DataFrame
    @assert( has_required_columns( a, ULLR_required_columns ), "Input dataframe does not have the required columns: $ULLR_required_columns. Please check spelling and capitalization of column names and ensure the format is ULLR." )
    @assert( has_required_columns( b, ULLR_required_columns ), "Input dataframe does not have the required columns: $ULLR_required_columns. Please check spelling and capitalization of column names and ensure the format is ULLR." )
    @assert( has_required_columns( a, [:Area] ), "Input dataframe does not have the required columns: Area. Please run `area()` on the individual dataframes." )
    @assert( has_required_columns( b, [:Area] ), "Input dataframe does not have the required columns: Area. Please run `area()` on the individual dataframes." )
    len_a, len_b = first.(size.( [ a, b ] ) )
    total_len = len_a * len_b

    output = DataFrame( (   :ID_A => Vector{Int}( undef, total_len ),
                            :ID_B => Vector{Int}( undef, total_len ),
                            :class_labels => Vector{Symbol}( undef, total_len ),
                            :Intersection_Area => Vector{Float64}( undef, total_len ),
                            :Intersection_Over_Union => Vector{Float64}( undef, total_len ),
                            :Vote_Matches => Vector{Bool}( undef, total_len ),
                            :Highest_Prediction => Vector{Float64}( undef, total_len),
                            :Highest_Vote => Vector{Symbol}( undef, total_len)
                        ) )
    current_row = 1
    for ( row_a_ID, row_a ) in enumerate( eachrow( a ) ), ( row_b_ID, row_b ) in enumerate( eachrow( b ) )
        #current_row = row_a_ID + ( len_a * ( row_b_ID - 1 ) )
        output[ current_row, :ID_A ] = row_a_ID
        output[ current_row, :ID_B ] = row_b_ID
        output[ current_row, :class_labels ] = row_b.class_labels
        output[ current_row, :Vote_Matches ] = row_a.Highest_Vote == row_b.class_labels
        output[ current_row, :Highest_Prediction ] = row_a.Highest_Prediction #row_b[!,row_a.Highest_Vote]
        output[ current_row, :Highest_Vote ] = row_a.Highest_Vote
        output[ current_row, :Intersection_Area ] = max( 0, min( row_a.lower_right_x, row_b.lower_right_x ) - max( row_a.upper_left_x, row_b.upper_left_x ) ) .*
                                                    max( 0, min( row_a.lower_right_y, row_b.lower_right_y ) - max( row_a.upper_left_y, row_b.upper_left_y ) )
        output[ current_row, :Intersection_Over_Union ] = output[ current_row, :Intersection_Area ] / ( row_a.Area + row_b.Area - output[ current_row, :Intersection_Area ])
        current_row += 1
    end
    #remove intermediate calculation column Intersection_Area
    select!(output, Not(:Intersection_Area))
    #Split data by each unique prediction
    #   -> sort by intersection over union (for each prediction all others are constant)
    #   -> grab the highest IoU and toss the rest.
    return combine( map( x -> sort(x, :Intersection_Over_Union, rev = true)[1,:], groupby( output, :ID_A ) ) )
end

mutable struct ObjectDetectionStats
    predictions::Vector{DataFrame}
    groundtruth::Vector{DataFrame}
    IoU_map::Vector{DataFrame}
    classnames::Vector{Symbol}
    len::Int
end


"""
    potential_scores( ods::ObjectDetectionStats; IoU_threshold = 0.5 )

"""
function potential_scores( cur_IoU_df::DataFrame; IoU_threshold = 0.5 )
    cur_IoU_df[ :iou_threshold ] = cur_IoU_df.Intersection_Over_Union .> IoU_threshold
    #is TP or FN?
    cur_IoU_df[ :TP_or_FN ] = cur_IoU_df.Vote_Matches .& cur_IoU_df.iou_threshold
    #Ensure we do not double dip from a GT prediction. Only one prediction can map to a GT here...
    #Group by ground truth class label
    #   -> sort by the highest predicted class score
    #       _-> find first prediced class matches above an IoU threshold
    for gdf in groupby( cur_IoU_df, :class_labels )
        sort!( cur_IoU_df, :Highest_Prediction, rev = true)
        kept_first = false
        for row in eachrow( gdf )
            if row.TP_or_FN
                if kept_first
                    row.TP_or_FN = false
                end
                kept_first = true
            end
        end
    end
    #Bifurcate DF on the possibility of true positives
    TP_or_FN, FP_or_FN = DataFrame.( collect( groupby( cur_IoU_df, :TP_or_FN ) ) )
    return TP_or_FN, FP_or_FN
end

"""
    ObjectDetectionStats(   predictions::Vector{DataFrame},
                            groundtruth::Vector{DataFrame},
                            classnames::Vector{Symbol} )::ObjectDetectionStats

Instantiates an `ObjectDetectionStats` object.

"""
function ObjectDetectionStats(  predictions::Vector{DataFrame},
                                groundtruth::Vector{DataFrame},
                                classnames::Vector{Symbol} )::ObjectDetectionStats
    len = length(predictions)
    @assert len == length(groundtruth) "Prediction and groundtruth Vector's must have the same length."
    area!( predictions )
    highest_vote!( predictions, classnames )
    area!( groundtruth )
    IoU_map = [ intersection_over_union( p, gt ) for (p, gt) in zip( predictions, groundtruth) ]
    return ObjectDetectionStats( predictions, groundtruth, IoU_map, classnames, len )
end



# each prediction maps to every GT
# predictions sorted by their confidence scores (highest to lowest)
# Each prediction can ONLY map to one GT
#  for each fixed prediction -> [GT] the prediction score is static
#  Need to find "most likely hit" via IoU.
#So sort by confidence and IoU
# Now for each group of predictions, remove everything but the top ranked ones
#Now we have the most representative prediction for each prediction to [GT]
#Find TP-or-FN mappings
#Find FP-or-FN mappings

using DataFrames
a = DataFrame(Dict([ :a => [1,1,1,2,2,2,3,3,3], :b => randn(9)]))
x,z,t = DataFrame.( collect( groupby(a, :a) ) )

a


KeyDF = DataFrame( Dict( [    :a               => [0.8, 0.1, 0.1],
                              :b               => [0.1, 0.7, 0.1],
                              :c               => [0.1, 0.2, 0.8],
                              :upper_left_x     => [12.5, 10.0 ,20.0],
                              :lower_right_x    => [17.5, 20.0, 30.0],
                              :upper_left_y     => [7.50, 5.00, 30.0],
                              :lower_right_y    => [12.5, 15.0, 35.0]
                         ] ) )

GT = DataFrame( Dict( [     :class_labels     => [:a, :b, :c],
                                :upper_left_x     => [12.5, 10.0, 16.0],
                                :lower_right_x    => [17.5, 20.0, 60.0],
                                :upper_left_y     => [7.50, 5.00, 16.0],
                                :lower_right_y    => [12.5, 15.0, 20.0]
                      ] ) )

ods = ObjectDetectionStats(  [KeyDF], [GT], [:a,:b,:c] )

score( ods, detection_threshold = 0.5, IoU_threshold = 0.5 )


@show ods.IoU_map[1]

#@benchmark join( KeyDF, KeyDFB, kind = :cross; makeunique = true )
#new_df = intersection_over_union( area( KeyDF ), area( KeyDFB ) )
@show( new_df )

# Pkg.add("DataFrames")
# using DataFrames
#
# b = DataFrame(  (:A => [1,2,1,2,2,2,3,3,3], :B => collect(1:9) )  )
#
# for gdf in groupby( b, :A )
#     gdf.B = map(x -> (x.B % 2) == 1, eachrow(gdf))
#     println(gdf)
# end
#
# b
