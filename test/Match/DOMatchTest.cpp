/*
 * =============================================================================
 *
 *       Filename:  main.cpp
 *
 *    Description:  Testing affine covariant region detectors.	
 *
 *        Version:  1.0
 *        Created:  16/06/2010 10:30:30
 *       Revision:  none
 *       Compiler:  msvc
 *
 *         Author:  David OK (DO), david.ok@imagine.enpc.fr 
 *        Company:  IMAGINE, (Ecole des Ponts ParisTech & CSTB)
 *
 * =============================================================================
 */

#include <DO/Match.hpp>

using namespace std;
using namespace DO;

int main()
{
	Match m(indexMatch(0, 1000));
  cout << m.indexPair().transpose() << endl;
	
	return 0;
}