#ifndef SERIALIZATION_HPP
#define SERIALIZATION_HPP

#include <iostream>

#include <boost/serialization/access.hpp>
#include <boost/serialization/nvp.hpp>

namespace boost{
namespace serialization{

template<   class archive,
            class S,
            int Rows_,
            int Cols_,
            int Ops_,
            int MaxRows_,
            int MaxCols_>
inline void serialize(archive & ar,
                      Eigen::Matrix<S, Rows_, Cols_, Ops_, MaxRows_, MaxCols_> & matrix,
                      const unsigned int version)
{
    int rows = matrix.rows();
    int cols = matrix.cols();
    ar & make_nvp("rows", rows);
    ar & make_nvp("cols", cols);
    matrix.resize(rows, cols); // no-op if size does not change!

    // always save/load row-major
    for(int r = 0; r < rows; ++r)
        for(int c = 0; c < cols; ++c)
            ar & make_nvp("val", matrix(r,c));
}

template<   class archive,
            class S,
            int Dim_,
            int Mode_,
            int Options_>
inline void serialize(archive & ar,
                      Eigen::Transform<S, Dim_, Mode_, Options_> & transform,
                      const unsigned int version)
{
    serialize(ar, transform.matrix(), version);
}

}
}

#endif //SERIALIZATION_HPP
