/**
 *  General definitions (implementation)
 *  @author Rodolfo Lima
 *  @author Andre Maximo
 *  @date February, 2012
 */

//== INCLUDES =================================================================

#include <defs.h>

//== NAMESPACES ===============================================================

namespace gpufilter {

//== IMPLEMENTATION ===========================================================

void calc_borders( int& left,
                   int& top,
                   int& right,
                   int& bottom,
                   const int& w,
                   const int& h,
                   const int& extb ) {

    left = extb*WS;
    top = extb*WS;

    if( extb > 0 ) {

        right = (extb+1)*WS-(w%WS);
        bottom = (extb+1)*WS-(h%WS);

    } else {

        right = WS-(w%WS);
        if( right == WS ) right = 0;
        bottom = WS-(h%WS);
        if( bottom == WS ) bottom = 0;

    }

}

bool extend( const int& w,
             const int& h,
             const int& extb ) {
    return (w%32>0 or h%32>0 or extb>0);
}

//=============================================================================
} // namespace gpufilter
//=============================================================================
