/******************************************************************************
 * Software License Agreement (BSD License)                                   *
 *                                                                            *
 * Copyright 2014 University of Utah                                          *
 * Scientific Computing and Imaging Institute                                 *
 * 72 S Central Campus Drive, Room 3750                                       *
 * Salt Lake City, UT 84112                                                   *
 *                                                                            *
 * THE BSD LICENSE                                                            *
 *                                                                            *
 * Redistribution and use in source and binary forms, with or without         *
 * modification, are permitted provided that the following conditions         *
 * are met:                                                                   *
 *                                                                            *
 * 1. Redistributions of source code must retain the above copyright          *
 *    notice, this list of conditions and the following disclaimer.           *
 * 2. Redistributions in binary form must reproduce the above copyright       *
 *    notice, this list of conditions and the following disclaimer in the     *
 *    documentation and/or other materials provided with the distribution.    *
 * 3. Neither the name of the copyright holder nor the names of its           *
 *    contributors may be used to endorse or promote products derived         *
 *    from this software without specific prior written permission.           *
 *                                                                            *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR       *
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES  *
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.    *
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,           *
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT   *
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,  *
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY      *
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT        *
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF   *
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.          *
 ******************************************************************************/

#include "GraphStructure.h"
#include "UnionFind.h"

#include <algorithm>
#include <utility>
#include <limits>
#include <cstdlib>

template<typename T>
void GraphStructure<T>::ComputeNeighborhood(std::vector<int> &edgeIndices,
                                            boost::numeric::ublas::matrix<int> &edges,
                                            boost::numeric::ublas::matrix<T> &dists,
                                            std::string type, T beta, int &kmax,
                                            bool connect)
{
  int numPts = Size();
  int dims = Dimension();

  T *pts = new T[numPts*dims];
  for(int i=0; i < numPts; i++)
    for(int d = 0; d < dims; d++)
      pts[i*dims+d] = X(d,i);

  ngl::Geometry<T>::init(dims);
  if(kmax<0)
    kmax = numPts-1;

  ngl::NGLPointSet<T> *P;
  ngl::NGLParams<T> params;
  params.param1 = beta;
  params.iparam0 = kmax;
  ngl::IndexType *indices = NULL;
  int numEdges = 0;

  if(edgeIndices.size() > 0)
  {
    P = new ngl::prebuiltNGLPointSet<T>(pts, numPts, edgeIndices);
  }
  else
  {
    P = new ngl::NGLPointSet<T>(pts, numPts);
  }

  std::map<std::string, graphFunction> graphAlgorithms;
  graphAlgorithms["approximate knn"]       = ngl::getKNNGraph<T>;
  graphAlgorithms["beta skeleton"]         = ngl::getBSkeleton<T>;
  graphAlgorithms["relaxed beta skeleton"] = ngl::getRelaxedBSkeleton<T>;
  //As it turns out, NGL's KNN graph assumes the input data is a KNN and so, is
  // actually just a pass through method that passes every input edge. We can
  // leverage this to accept "none" graphs.
  graphAlgorithms["none"]                  = ngl::getKNNGraph<T>;

  if(graphAlgorithms.find(type) == graphAlgorithms.end())
  {
    //TODO
    //These checks can probably be done upfront, so as not to waste computation
    std::cerr << "Invalid graph type: " << type << std::endl;
    exit(1);
  }

  graphAlgorithms[type](*P,&indices,numEdges,params);

  std::stringstream ss;
  ss << "\t\t(Edges: " << numEdges << ")" << std::endl;

  delete [] pts;
  delete P;

  if(connect)
  {
    std::set< std::pair<int,int> > ngraph;
    for(int i = 0; i < numEdges; i++)
    {
      std::pair<int,int> edge;
      if(indices[2*i+0] > indices[2*i+1])
      {
        edge.first = indices[2*i+1];
        edge.second = indices[2*i+0];
      }
      else
      {
        edge.first = indices[2*i+0];
        edge.second = indices[2*i+1];
      }
      ngraph.insert(edge);
    }

    ConnectComponents(ngraph, kmax);
    //    edges.deallocate();
    //    dists.deallocate();
    edges = boost::numeric::ublas::matrix<int>(kmax,Size());
    dists = boost::numeric::ublas::matrix<T>(kmax,Size());

    boost::numeric::ublas::vector<int> nextNeighborId(numPts);
    for(int i = 0; i < numPts; i++)
    {
      nextNeighborId(i) = 0;
      for(int k = 0; k < kmax; k++)
      {
        edges(k,i) = -1;
        dists(k,i) = -1;
      }
    }
    for(std::set< std::pair<int,int> >::iterator it = ngraph.begin();
        it != ngraph.end(); it++)
    {
      int i1 = it->first;
      int i2 = it->second;
      double dist = 0;
      for(int d = 0; d < dims; d++)
          dist += ((X(d,i1)-X(d,i2))*(X(d,i1)-X(d,i2)));

      int j = nextNeighborId(i1);
      nextNeighborId(i1) = nextNeighborId(i1) + 1;
      edges(j, i1) = i2;
      dists(j,i1) = dist;

      j = nextNeighborId(i2);
      nextNeighborId(i2) = nextNeighborId(i2) + 1;
      edges(j, i2) = i1;
      dists(j,i2) = dist;
    }
  }
  else
  {
    int *neighborCounts = new int[numPts];
    for(int i =0; i < numPts; i++)
      neighborCounts[i] = 0;

    for(int i =0; i < numEdges; i++)
    {
      int i1 = indices[2*i+0];
      int i2 = indices[2*i+1];
      neighborCounts[i1]++;
      neighborCounts[i2]++;
    }

    for(int i =0; i < numPts; i++)
      kmax = neighborCounts[i] > kmax ? neighborCounts[i] : kmax;
    delete [] neighborCounts;

    edges = boost::numeric::ublas::matrix<int>(kmax,Size());
    dists = boost::numeric::ublas::matrix<T>(kmax,Size());

    boost::numeric::ublas::vector<int> nextNeighborId(numPts);
    for(int i = 0; i < numPts; i++)
    {
      nextNeighborId(i) = 0;
      for(int k = 0; k < kmax; k++)
      {
        edges(k,i) = -1;
        dists(k,i) = -1;
      }
    }

    for(int i = 0; i < numEdges; i++)
    {
      int i1 = indices[2*i+0];
      int i2 = indices[2*i+1];
      if(i1 > i2)
      {
        int temp = i2;
        i2 = i1;
        i1 = temp;
      }

      double dist = 0;
      for(int d = 0; d < dims; d++)
          dist += ((X(d,i1)-X(d,i2))*(X(d,i1)-X(d,i2)));

      int j = nextNeighborId(i1);
      nextNeighborId(i1) = nextNeighborId(i1) + 1;
      edges(j, i1) = i2;
      dists(j,i1) = dist;

      j = nextNeighborId(i2);
      nextNeighborId(i2) = nextNeighborId(i2) + 1;
      edges(j, i2) = i1;
      dists(j,i2) = dist;
    }
  }

  for(int i = 0; i < numPts; i++)
    //TODO: too many neighborhood representations floating around, this one is
    //      useful for later queries to the data, when the user wants to ask
    //      who is near point x?
    neighbors[i] = std::set<int>();
  for(int i = 0; i < numPts; i++)
  {
    for(int k = 0; k < kmax; k++)
    {
      //TODO: too many neighborhood representations floating around, this one is
      //      useful for later queries to the data, when the user wants to ask
      //      who is near point x?
      int i2 = edges(k,i);
      if(i2 != -1)
      {
        neighbors[i].insert(i2);
        neighbors[i2].insert(i);
      }
    }
  }
}

template<typename T>
GraphStructure<T>::GraphStructure(std::vector<T> &Xin, int rows, int cols,
                                  std::string graph, int maxN, T beta,
                                  std::vector<int> &edgeIndices)
{
  // This boolean flag dictates whether the dataset should be forced to be a
  // single connected component. This feature might get deprecated or promoted
  // to be exposed to the user, for now I will enforce that it does not happen
  bool connect = false;

  int M = cols;
  int N = rows;

  X = boost::numeric::ublas::matrix<T>(M,N);

  for(int n = 0; n < N; n++)
  {
    for(int m = 0; m < M; m++)
    {
      X(m,n) = Xin[n*M+m];
    }
  }

  boost::numeric::ublas::matrix<int> edges;
  boost::numeric::ublas::matrix<T> distances;
  int kmax = maxN;

  ComputeNeighborhood(edgeIndices, edges, distances, graph, beta, kmax,connect);
 // edges.deallocate();
}

template<typename T>
void GraphStructure<T>::ConnectComponents(std::set<int_pair> &ngraph,
                                          int &maxCount)
{
  UnionFind connectedComponents;
  for(int i = 0; i < Size(); i++)
    connectedComponents.MakeSet(i);

  for(std::set<int_pair>::iterator iter= ngraph.begin();
      iter != ngraph.end();
      iter++)
  {
    connectedComponents.Union(iter->first,iter->second);
  }

  int numComponents = connectedComponents.CountComponents();
  std::vector<int> reps;
  connectedComponents.GetComponentRepresentatives(reps);
  if(numComponents > 1)
  {
    std::stringstream ss;
    ss << "Connected Components: " << numComponents << "(Graph size: "
       << ngraph.size() << ")" << std::endl;
    for(unsigned int i = 0; i < reps.size(); i++)
      ss << reps[i] << " ";
  }

  while(numComponents > 1)
  {
    //Get each representative of a component and store each
    // component into its own set
    std::vector<int> reps;
    connectedComponents.GetComponentRepresentatives(reps);
    std::vector<int> *components = new std::vector<int>[reps.size()];
    for(unsigned int i = 0; i < reps.size(); i++)
      connectedComponents.GetComponentItems(reps[i],components[i]);

    //Determine closest points between all pairs of components
    double minDistance = -1;
    int p1 = -1;
    int p2 = -1;

    for(unsigned int a = 0; a < reps.size(); a++)
    {
      for(unsigned int b = a+1; b < reps.size(); b++)
      {
        for(unsigned int i = 0; i < components[a].size(); i++)
        {
          int AvIdx = components[a][i];
          std::vector<T> ai;
          for(int d = 0; d < Dimension(); d++)
              ai.push_back(X(d,AvIdx));
          for(unsigned int j = 0; j < components[b].size(); j++)
          {
            int BvIdx = components[b][j];
            std::vector<T> bj;
            for(int d = 0; d < Dimension(); d++)
              bj.push_back(X(d,BvIdx));

            T distance = 0;
            for(int d = 0; d < Dimension(); d++)
              distance += (ai[d]-bj[d])*(ai[d]-bj[d]);
            if(minDistance == -1 || distance < minDistance)
            {
              minDistance = distance;
              p1 = components[a][i];
              p2 = components[b][j];
            }
          }
        }
      }
    }

    //Merge
    connectedComponents.Union(p1,p2);
    if(p1 < p2)
    {
      int_pair edge = std::make_pair(p1,p2);
      ngraph.insert(edge);
    }
    else
    {
      int_pair edge = std::make_pair(p1,p2);
      ngraph.insert(edge);
    }

    //Recompute
    numComponents = connectedComponents.CountComponents();
    if(numComponents > 1)
    {
      std::stringstream ss;
      ss << "Connected Components: " << numComponents << "(Graph size: "
         << ngraph.size() << ")" << std::endl;
    }

    delete [] components;
  }
  int *counts = new int[Size()];
  for(int i = 0; i < Size(); i++)
    counts[i] = 0;

  for(std::set<int_pair>::iterator it = ngraph.begin();
      it != ngraph.end();
      it++)
  {
    counts[it->first]+=1;
    counts[it->second]+=1;
  }
  for(int i = 0; i < Size(); i++)
    maxCount = maxCount < counts[i] ? counts[i] : maxCount;

  delete [] counts;
}

//Look-up Operations

template<typename T>
int GraphStructure<T>::Dimension()
{
//  return (int)X.M();
  return (int) X.size1();
}

template<typename T>
int GraphStructure<T>::Size()
{
//  return (int) X.N();
  return (int) X.size2();
}

template<typename T>
void GraphStructure<T>::GetX(int i, T *xi)
{
  for(int d = 0; d < Dimension(); d++)
    xi[d] = X(d,i);
}

template<typename T>
T GraphStructure<T>::GetX(int i, int j)
{
  return X(i,j);
}

template<typename T>
T GraphStructure<T>::MinX(int dim)
{
  T minX = X(dim,0);
  for(int i = 1; i < Size(); i++)
    minX = minX > X(dim,i) ? X(dim,i) : minX;
  return minX;
}

template<typename T>
T GraphStructure<T>::MaxX(int dim)
{
  T maxX = X(dim,0);
  for(int i = 1; i < Size(); i++)
    maxX = maxX < X(dim,i) ? X(dim,i) : maxX;
  return maxX;
}

template<typename T>
T GraphStructure<T>::RangeX(int dim)
{
  return MaxX(dim)-MinX(dim);
}

template<typename T>
std::set<int> GraphStructure<T>::Neighbors(int index)
{
  return neighbors[index];
}

template class GraphStructure<double>;
template class GraphStructure<float>;