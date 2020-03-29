mutable struct Box
    upper_left_x::Number
    upper_left_y::Number
    lower_right_x::Number
    lower_right_y::Number
end


"""
    Base.:(==)(a::Box, b::Box)::Bool
    
Tests for equivalence between two Box instances.
"""
function Base.:(==)(a::Box, b::Box)::Bool
    return  (a.upper_left_x == b.upper_left_x) &&
            (a.upper_left_y == b.upper_left_y) &&
            (a.lower_right_x == b.lower_right_x) &&
            (a.lower_right_y == b.lower_right_y)
end

"""
    barycenterbox( x::Number, y::Number, height::Number, width::Number)

Converts a detection box from x,y,h,w coordinates to upper left lower right
coordinates.
"""
function barycenterbox( x::Number, y::Number, height::Number, width::Number)
    wh, hh = width/2, height/2
    return Box( x - wh, y - hh, x + wh, y + hh )
end

"""
    LLURtoULLR( LL_x::Number, LL_y::Number, UR_x::Number, UR_y::Number)

Converts a detection box from lowerleft-upperright coordinates to upper left lower right
coordinates.

Untested...
"""
function LLURtoULLR( LL_x::Number, LL_y::Number, UR_x::Number, UR_y::Number)
    return Box( LL_x, UR_y, UR_x, LL_y )
end

"""
    area( a::Box )

Calculates the area of a Box instance.

"""
area( a::Box ) = ( a.lower_right_x - a.upper_left_x ) * ( a.lower_right_y - a.upper_left_y )

"""
    translate(a::Box, x::Number,y::Number)::Box

translates a box by the input x & y coordinates.
"""
function translate(a::Box, x::Number = 0, y::Number = 0)::Box
    a.upper_left_x  += x
    a.upper_left_y  += y
    a.lower_right_x += x
    a.lower_right_y += y
    return a
end


"""
    clamp(a::Box, bound::Box)::Box

Clamps a `Box` instance `a` to be within a bounded `Box` region `bound`.

"""
function clamp(a::Box, bound::Box)::Box
    a.upper_left_x  = (a.upper_left_x < bound.upper_left_x) ? bound.upper_left_x : a.upper_left_x
    a.upper_left_y  = (a.upper_left_y < bound.upper_left_y) ? bound.upper_left_y : a.upper_left_y
    a.lower_right_x = (a.lower_right_x > bound.lower_right_x) ? bound.lower_right_x : a.lower_right_x
    a.lower_right_y = (a.lower_right_y > bound.lower_right_y) ? bound.lower_right_y : a.lower_right_y
    return a
end

"""
    intersection_area( a::Box, b::Box )::Number

Calculates the area of intersection between 2 Box instances.

"""
function intersection_area( a::Box, b::Box )::Number
	ul_x = max( a.upper_left_x,  b.upper_left_x )
	ul_y = max( a.upper_left_y,  b.upper_left_y )
	lr_x = min( a.lower_right_x, b.lower_right_x )
	lr_y = min( a.lower_right_y, b.lower_right_y )
	return max( 0, lr_x - ul_x ) * max( 0, lr_y - ul_y )
end

"""
    intersection_over_union( a::Box, b::Box )::Number

Calculates the intersection over union between 2 Box instances.

"""
function intersection_over_union( a::Box, b::Box )::Number
    intersection = intersection_area( a, b )
    return intersection / ( area( a ) + area( b ) - intersection)
end
