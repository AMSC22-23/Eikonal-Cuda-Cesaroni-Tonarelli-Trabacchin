/*
 * DescentDirections.hpp
 *
 *  Created on: Dec 28, 2020
 *      Author: forma
 */

#ifndef EXAMPLES_SRC_LINESEARCH_DESCENTDIRECTIONS_HPP_
#define EXAMPLES_SRC_LINESEARCH_DESCENTDIRECTIONS_HPP_
#include "DescentDirectionBase.hpp"
#include "Eigen/Dense"
#include <limits>
#include <numeric>
#include <functional>
#include <array>
namespace apsc
{
/*!
 * Implements the gradient search
 */
    template<int N>
    class GradientDirection : public DescentDirectionBase<N>
    {
    public:
        /*!
         *  Returns - gradient as descent direction
         * @param values The current values.
         * @return The descent direction.
         */
        typename apsc::LineSearch_traits<N>::Vector
        operator()(apsc::OptimizationCurrentValues<N> const &values) override
        {
            return -values.currentGradient;
        }
        /*!
         * @brief The class is clonable
         *
         * @return A clone of myself wrapped into a unique pointer
         */
        virtual
        std::unique_ptr<DescentDirectionBase<N>>
        clone() const override
        {return std::make_unique<GradientDirection>(*this);}

    };
/*!
 * Implements the gradient search
 */
    template<int N>
    class NewtonDirection : public DescentDirectionBase<N>
    {
    public:
        /*!
         *  Returns - gradient as descent direction
         * @param values The current values.
         * @return The descent direction.
         */
        typename apsc::LineSearch_traits<N>::Vector
        operator()(apsc::OptimizationCurrentValues<N> const &values) override
        {
            //    if(!values.bounded)
            //      return values.currentHessian.llt().solve(-values.currentGradient);
            //    else
            //      {
            if constexpr (N == 2) {
                bool active=false;
                std::array<bool,3> constrained={false,false,false};
                for (auto i=0; i<values.currentPoint.size();++i)
                {
                    constrained[i]=
                            (values.currentPoint[i]==0.0
                             and values.currentGradient[i]>0 );
                    active= active || constrained[i];
                }

                constrained[2]=(std::abs(values.currentPoint[0]+values.currentPoint[1]-1)<=eps)
                               and (values.currentGradient[0]+values.currentGradient[1])<=0.0;
                active=active||constrained[2];
                if(not active)
                    return -values.currentHessian.inverse()*values.currentGradient;
                if(
                        (constrained[0] and constrained[1])
                        or
                        (values.currentPoint[0]==1. and values.currentPoint[1]==0. and
                         (values.currentGradient[1]-values.currentGradient[0])>=0. and
                         values.currentGradient[0]<=0.)
                        or
                        (values.currentPoint[0]==0. and values.currentPoint[1]==1. and
                         (values.currentGradient[1]-values.currentGradient[0])<=0. and
                         values.currentGradient[1]<=0.)
                        )
                {
                    // gradient is pushing outside the constrained area
                    return ZeroVec;
                }

                typename apsc::LineSearch_traits<N>::Matrix Hi =values.currentHessian.inverse();
                if(constrained[0])
                {
                    Hi.row(0).fill(0.);
                    Hi.col(0).fill(0.);
                }
                else if	(constrained[1])
                {
                    Hi.row(1).fill(0.);
                    Hi.col(1).fill(0.);
                }
                else if(constrained[2])
                {
                    Hi=P3*Hi*P3;
                }
                return -Hi*values.currentGradient;
            } else {
                bool active = ((values.currentPoint[0] == 0.0
                                and values.currentGradient[0] > 0) or
                               (values.currentPoint[0] == 1.0
                                and values.currentGradient[0] < 0));
                if (active)
                    return ZeroVec;
                else
                    return -values.currentHessian.inverse() * values.currentGradient;
            }

        }

/*!
 * @brief The class is clonable
 *
 * @return A clone of myself wrapped into a unique pointer
 */
        virtual
        std::unique_ptr<DescentDirectionBase<N>>
        clone() const override {return std::make_unique<NewtonDirection>(*this);}

    private:
        static inline const typename apsc::LineSearch_traits<N>::Vector ZeroVec{apsc::LineSearch_traits<N>::Vector::Zero()};
        static inline const typename apsc::LineSearch_traits<N>::Matrix P3{apsc::LineSearch_traits<N>::Matrix::Identity()-
                                                                           0.5*apsc::LineSearch_traits<N>::Matrix::Ones()};
        static constexpr double eps=100.*std::numeric_limits<double>::epsilon();
    };
/*!
 *  Implements the classic BFGS quasi-Newton algorithm.
 */
    template<int N>
    class BFGSDirection : public DescentDirectionBase<N>
    {
    public:
        typename apsc::LineSearch_traits<N>::Vector
        operator()(apsc::OptimizationCurrentValues<N> const &values) override;
        /*!
         * You need to reset if you run another problem or you start from a different
         * initial point.
         * @note This is done inside LineSearchSolver
         */
        void reset() override;
        /*!
          * @brief The class is clonable
          *
          * @return A clone of myself wrapped into a unique pointer
          */
        virtual
        std::unique_ptr<DescentDirectionBase<N>>
        clone() const override
        {return std::make_unique<BFGSDirection>(*this);}


    private:
        apsc::OptimizationCurrentValues<N> previousValues;
        Eigen::MatrixXd                 H;
        bool                            firstTime{true};
        double const smallNumber = std::sqrt(std::numeric_limits<double>::epsilon());
    };

/*!
 * Implements BFGS with the direct computation of the approximate inverse of
 * the Hessian.
 */
    template<int N>
    class BFGSIDirection : public DescentDirectionBase<N>
    {
    public:
        typename apsc::LineSearch_traits<N>::Vector
        operator()(apsc::OptimizationCurrentValues<N> const &values) override;
        void reset() override;

    private:
        apsc::OptimizationCurrentValues<N> previousValues;
        Eigen::MatrixXd                 H;
        bool                            firstTime{true};
        double const smallNumber = std::sqrt(std::numeric_limits<double>::epsilon());

        /*!
          * @brief The class is clonable
          *
          * @return A clone of myself wrapped into a unique pointer
          */
        virtual
        std::unique_ptr<DescentDirectionBase<N>>
        clone() const override
        {return std::make_unique<BFGSIDirection>(*this);}


    };
/*!
 * Bazrzilain-Borwein
 */
    template<int N>
    class BBDirection : public DescentDirectionBase<N>
    {
    public:
        typename apsc::LineSearch_traits<N>::Vector
        operator()(apsc::OptimizationCurrentValues<N> const &values) override;
        void
        reset() override
        {
            firstTime = true;
        };
        /*!
          * @brief The class is clonable
          *
          * @return A clone of myself wrapped into a unique pointer
          */
        virtual
        std::unique_ptr<DescentDirectionBase<N>>
        clone() const override
        {return std::make_unique<BBDirection>(*this);}


    private:
        apsc::OptimizationCurrentValues<N> previousValues;
        bool                            firstTime{true};
        double const smallNumber = std::sqrt(std::numeric_limits<double>::epsilon());
    };
/*!
 * Non-linear CG with Polak-Ribiere formula
 */
    template<int N>
    class CGDirection : public DescentDirectionBase<N>
    {
    public:
        typename apsc::LineSearch_traits<N>::Vector
        operator()(apsc::OptimizationCurrentValues<N> const &values) override;
        void
        reset() override
        {
            firstTime = true;
        };

        /*!
          * @brief The class is clonable
          *
          * @return A clone of myself wrapped into a unique pointer
          */
        virtual
        std::unique_ptr<DescentDirectionBase<N>>
        clone() const override
        {return std::make_unique<CGDirection>(*this);}


    private:
        apsc::OptimizationCurrentValues<N> previousValues;
        bool                            firstTime{true};
        //! I need to keep track of previous descent direction
        typename apsc::LineSearch_traits<N>::Vector prevDk;
    };









    template<int N>
    typename apsc::LineSearch_traits<N>::Vector
    apsc::BFGSDirection<N>::operator()(const apsc::OptimizationCurrentValues<N> &values)
    {
        // First time is a gradient step
        if(firstTime)
        {
            auto const n = values.currentPoint.size();
            H = Eigen::MatrixXd::Identity(n, n);
            firstTime = false;
            this->previousValues = values;
            return -values.currentGradient;
        }

        typename apsc::LineSearch_traits<N>::Vector const &g = values.currentGradient;
        typename apsc::LineSearch_traits<N>::Vector yk = g - this->previousValues.currentGradient;
        typename apsc::LineSearch_traits<N>::Vector sk =
                values.currentPoint - this->previousValues.currentPoint;
        auto const yks = yk.dot(sk);
        // Correct approximate Hessian only if we maintain sdp property if not keep
        // the old one
        if(yks > this->smallNumber * sk.norm() * yk.norm())
        {
            typename apsc::LineSearch_traits<N>::Vector Hs;
            Hs = H * sk;
            H += (yk * yk.transpose()) / yks - (Hs * Hs.transpose()) / (sk.dot(Hs));
        }
        this->previousValues = values;
        typename apsc::LineSearch_traits<N>::Vector d = H.fullPivLu().solve(-g);
        return d;
    }

    template<int N>
    void
    apsc::BFGSDirection<N>::reset()
    {
        this->firstTime = true;
    }

/*                   BFGS with approximate inverse */
    template<int N>
    typename apsc::LineSearch_traits<N>::Vector
    apsc::BFGSIDirection<N>::operator()(const apsc::OptimizationCurrentValues<N> &values)
    {
        // First time is a gradient step
        if(firstTime)
        {
            auto n = values.currentPoint.size();
            H = Eigen::MatrixXd::Identity(n, n);
            firstTime = false;
            this->previousValues = values;
            return -values.currentGradient;
        }

        typename apsc::LineSearch_traits<N>::Vector const &g = values.currentGradient;
        typename apsc::LineSearch_traits<N>::Vector yk = g - this->previousValues.currentGradient;
        typename apsc::LineSearch_traits<N>::Vector sk =
                values.currentPoint - this->previousValues.currentPoint;
        auto const yks = yk.dot(sk);
        // Correct approximate Hessian only if we maintain sdp property if not keep
        // the old one
        if(yks > this->smallNumber * sk.norm() * yk.norm())
        {
            H += sk * sk.transpose() * (yks + yk.transpose() * H * yk) / (yks * yks) -
                 (H * yk * sk.transpose() + sk * yk.transpose() * H) / yks;
        }
        this->previousValues = values;
        typename apsc::LineSearch_traits<N>::Vector d = -H * g;
        return d;
    }

    template<int N>
    void
    apsc::BFGSIDirection<N>::reset()
    {
        this->firstTime = true;
    }

/*
 * Barzilai-Borwain
 */
    template<int N>
    typename apsc::LineSearch_traits<N>::Vector
    apsc::BBDirection<N>::operator()(const apsc::OptimizationCurrentValues<N> &values)
    {
        // First time is a gradient step
        if(firstTime)
        {
            firstTime = false;
            this->previousValues = values;
            return -values.currentGradient;
        }

        typename apsc::LineSearch_traits<N>::Vector const &g = values.currentGradient;
        typename apsc::LineSearch_traits<N>::Vector yk = g - this->previousValues.currentGradient;
        typename apsc::LineSearch_traits<N>::Vector sk =
                values.currentPoint - this->previousValues.currentPoint;
        auto yks = yk.dot(sk);
        auto ykk = yk.dot(yk);
        this->previousValues = values;
        // Correct approximate Hessian only if we maintain sdp property if not keep
        // the old one
        if(yks > this->smallNumber * sk.norm() * yk.norm() and ykk > smallNumber)
        {
            // I use a mix of the two possible strategies
            return -0.5 * (yks / ykk + sk.dot(sk) / yks) * g;
            // Strategy 1
            // return -(yks/ykk)*g;
            // Strategy 2
            // return -(sk.dot(sk)/yks)*g;
        }
        else
        {
            return -g;
        }
    }

/*
 * Non-linear CG with Polak-Ribier formula
 */
    template<int N>
    typename apsc::LineSearch_traits<N>::Vector
    apsc::CGDirection<N>::operator()(const apsc::OptimizationCurrentValues<N> &values)
    {
        // First time is a gradient step
        if(firstTime)
        {
            firstTime = false;
            this->prevDk = -values.currentGradient;
        }
        else
        {
            typename apsc::LineSearch_traits<N>::Vector const &gk1 = values.currentGradient;
            typename apsc::LineSearch_traits<N>::Vector const &gk =
                    this->previousValues.currentGradient;
            typename apsc::LineSearch_traits<N>::Vector const &dk = this->prevDk;
            // Polak Ribiere formula
            this->prevDk =
                    -gk1 + (gk1.dot(gk1 - gk) / (gk.dot(gk))) * dk; // store for next time
            if(prevDk.dot(gk1) > 0)
                prevDk =
                        -gk1; // check if direction is a descent if not go back to gradient
        }
        this->previousValues = values; // store for next time
        return this->prevDk;
    }









} // namespace apsc

#endif /* EXAMPLES_SRC_LINESEARCH_DESCENTDIRECTIONS_HPP_ */
