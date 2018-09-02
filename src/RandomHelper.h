#ifndef GRAVITYPOSE_RANDOMHELPER_H
#define GRAVITYPOSE_RANDOMHELPER_H

#include <chrono>
#include <random>

template <typename T>
class GaussianGenerator
//===================
{
public:
   explicit GaussianGenerator(T _deviation, T _mean =0, std::mt19937* _gen = nullptr) : deviation(_deviation)
   //--------------------------------------------------------------------------------------------------------
   {
      mean = _mean;
      if (_gen == nullptr)
      {
         long seed = std::chrono::system_clock::now().time_since_epoch().count();
         generator = new std::mt19937(seed);
         is_own_generator = true;
      }
      else
      {
         generator = _gen;
         is_own_generator = false;
      }
   }

   ~GaussianGenerator()
   //-------------------------
   {
      if ( (is_own_generator) && (generator != nullptr) )
         delete generator;
   }

   T operator()()
   //-----------------
   {
      std::normal_distribution<T> N(mean, deviation);
      return N(*generator);
   }

   T g()
   {
      std::normal_distribution<T> N(mean, deviation);
      return N(*generator);
   }


   template <typename ...TT>
   void many(T& first, TT&... rest)
   //--------------
   {
      std::normal_distribution<T> N(mean, deviation);
      first = N(*generator);
      many(N, rest ...);
   }

private:
   void many(std::normal_distribution<T>& N, T& first) {  first = N(*generator); }

   T mean, deviation;
   std::mt19937 *generator = nullptr;
   bool is_own_generator = false;
};

template <typename T>
class UniformGenerator
//===================
{
public:
   explicit UniformGenerator(T _max =1, T _min =0, std::mt19937* _gen = nullptr) : min(_min), max(_max)
   //---------------------------------------------------------------------------------------------------
   {
      if (_gen == nullptr)
      {
         long seed = std::chrono::system_clock::now().time_since_epoch().count();
         generator = new std::mt19937(seed);
         is_own_generator = true;
      }
      else
      {
         generator = _gen;
         is_own_generator = false;
      }
   }

   ~UniformGenerator()
   //-------------------------
   {
      if ( (is_own_generator) && (generator != nullptr) )
         delete generator;
   }

   T operator()()
   //-----------------
   {
      std::uniform_real_distribution<T> U(min, max);
      return U(*generator);
   }

   T g()
   {
      std::uniform_real_distribution<T> U(min, max);
      return U(*generator);
   }


   template <typename ...TT>
   void many(T& first, TT&... rest)
   //--------------
   {
      std::uniform_real_distribution<T> U(min, max);
      first = U(*generator);
      many(U, rest ...);
   }

private:
   void many(std::uniform_real_distribution<T>& U, T& first) {  first = U(*generator); }

   T min, max;
   std::mt19937 *generator = nullptr;
   bool is_own_generator = false;
};


#endif //GRAVITYPOSE_RANDOMHELPER_H
