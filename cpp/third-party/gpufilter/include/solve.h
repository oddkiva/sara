/**
 *  @file solve.h
 *  @brief Definitions for tridiagonal spline solve
 *  @author Andre Maximo
 *  @date Dec, 2012
 */

#ifndef SOLVE_H
#define SOLVE_H

namespace spline {

const float l[] = { 0.2f, 0.26315789f, 0.26760563f, 0.26792453f,
                    0.26794742f, 0.26794907f, 0.26794918f, 0.26794919f };
const int ln = sizeof(l)/sizeof(l[0]);
const float p = 1/6.f, q = 4*p, v = 4.73205078f;
const float linf = l[ln-1], pinv = 1/p, vinv = 1/v, pinv_v = pinv*vinv;

// unser_etal_pami1991 (3.13) and (3.16) define a cascaded (sum) and
// parallel (mul) solutions for recursive filtering
const float alpha = sqrt(3)-2, mb0=-6*alpha, sb0 = mb0/(1-alpha*alpha), b1=alpha;

// nehab_etal_tog2011 (introduction) and (1) define recursive
// operation with subtraction of previous elements
const float w0=sqrt(mb0), w1=-b1;

enum solve_type {
    traditional = 0, unser313, unser316, unser318, nehab6
};

// unser_etal_pami1991 (3.13)
template <class T>
void unser_etal_pami1991_3_13( T *inout,
                               const int& w, const int& h ) {
    for (int y=0; y<h; ++y) {
        for (int x=1; x<w; ++x)
            inout[y*w+x] += b1*inout[y*w+x-1];
        for (int x=w-2; x>=0; --x)
            inout[y*w+x] += b1*inout[y*w+x+1];
        for (int x=0; x<w; ++x)
            inout[y*w+x] *= mb0;
    }
    for (int x=0; x<w; ++x) {
        for (int y=1; y<h; ++y)
            inout[y*w+x] += b1*inout[(y-1)*w+x];
        for (int y=h-2; y>=0; --y)
            inout[y*w+x] += b1*inout[(y+1)*w+x];
        for (int y=0; y<h; ++y)
            inout[y*w+x] *= mb0;
    }
}

// unser_etal_pami1991 (3.16)
template <class T>
void unser_etal_pami1991_3_16( T *inout,
                               const int& w, const int& h ) {
    T *cplus = new T[w*h], *cminus = new T[w*h];
    for (int y=0; y<h; ++y) {
        cplus[y*w+0] = inout[y*w+0];
        for (int x=1; x<w; ++x)
            cplus[y*w+x] = inout[y*w+x]+b1*cplus[y*w+x-1];
        cminus[y*w+w-1] = inout[y*w+w-1];
        for (int x=w-2; x>=0; --x)
            cminus[y*w+x] = inout[y*w+x]+b1*cminus[y*w+x+1];
        for (int x=0; x<w; ++x)
            inout[y*w+x] = sb0*(cplus[y*w+x]+cminus[y*w+x]-inout[y*w+x]);
    }
    for (int x=0; x<w; ++x) {
        cplus[x] = inout[x];
        for (int y=1; y<h; ++y)
            cplus[y*w+x] = inout[y*w+x]+b1*cplus[(y-1)*w+x];
        cminus[(h-1)*w+x] = inout[(h-1)*w+x];
        for (int y=h-2; y>=0; --y)
            cminus[y*w+x] = inout[y*w+x]+b1*cminus[(y+1)*w+x];
        for (int y=0; y<h; ++y)
            inout[y*w+x] = sb0*(cplus[y*w+x]+cminus[y*w+x]-inout[y*w+x]);
    }
    delete [] cplus;
    delete [] cminus;
}

// unser_etal_pami1991 (3.18)
//   it is not minus alpha in the first equation
template <class T>
void unser_etal_pami1991_3_18( T *inout,
                               const int& w, const int& h ) {
    T *dplus = new T[w*h];
    for (int y=0; y<h; ++y) {
        dplus[y*w+0] = 6*inout[y*w+0];
        for (int x=1; x<w; ++x)
            dplus[y*w+x] = 6*inout[y*w+x]+alpha*dplus[y*w+x-1];
        inout[y*w+w-1] = (-alpha/(1-alpha*alpha))*(2*dplus[y*w+w-1]-6*inout[y*w+w-1]);
        for (int x=w-2; x>=0; --x)
            inout[y*w+x] = alpha*(inout[y*w+x+1]-dplus[y*w+x]);
    }
    for (int x=0; x<w; ++x) {
        dplus[x] = 6*inout[x];
        for (int y=1; y<h; ++y)
            dplus[y*w+x] = 6*inout[y*w+x]+alpha*dplus[(y-1)*w+x];
        inout[(h-1)*w+x] = (-alpha/(1-alpha*alpha))*(2*dplus[(h-1)*w+x]-6*inout[(h-1)*w+x]);
        for (int y=h-2; y>=0; --y)
            inout[y*w+x] = alpha*(inout[(y+1)*w+x]-dplus[y*w+x]);
    }
    delete [] dplus;
}

// nehab_hoppe_tr2011 sec.6
//   it solves the reflect problem
template <class T>
void nehab_hoppe_tr2011_sec6( T *inout,
                              const int& w, const int& h ) {
    for (int y=0; y<h; ++y) {
        for (int x=1; x<=ln; ++x)
            inout[y*w+x] -= l[x-1]*inout[y*w+x-1];
        for (int x=ln+1; x<w; ++x)
            inout[y*w+x] -= linf*inout[y*w+x-1];
        inout[y*w+w-1] = pinv_v*inout[y*w+w-1];
        for (int x=w-2; x>=ln; --x)
            inout[y*w+x] = linf*(pinv*inout[y*w+x]-inout[y*w+x+1]);
        for (int x=ln-1; x>=0; --x)
            inout[y*w+x] = l[x]*(pinv*inout[y*w+x]-inout[y*w+x+1]);
    }
    for (int x=0; x<w; ++x) {
        for (int y=1; y<=ln; ++y)
            inout[y*w+x] -= l[y-1]*inout[(y-1)*w+x];
        for (int y=ln+1; y<h; ++y)
            inout[y*w+x] -= linf*inout[(y-1)*w+x];
        inout[(h-1)*w+x] = pinv_v*inout[(h-1)*w+x];
        for (int y=h-2; y>=ln; --y)
            inout[y*w+x] = linf*(pinv*inout[y*w+x]-inout[(y+1)*w+x]);
        for (int y=ln-1; y>=0; --y)
            inout[y*w+x] = l[y]*(pinv*inout[y*w+x]-inout[(y+1)*w+x]);
    }
}

}

#endif // SOLVE_H

