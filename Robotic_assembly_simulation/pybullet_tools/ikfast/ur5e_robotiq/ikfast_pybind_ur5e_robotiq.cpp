#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// #include <pybind11/eigen.h>
// #include <pybind11/iostream.h>

#ifndef IKFAST_HAS_LIBRARY
#define IKFAST_HAS_LIBRARY
#endif

#include "ikfast.h"

#ifdef IKFAST_NAMESPACE
using namespace IKFAST_NAMESPACE;
#endif

namespace py = pybind11;

PYBIND11_MODULE(ikfast_ur5e_robotiq, m)
{
    m.doc() = R"pbdoc(
        ur5e_robotiq
        -----------------------
        .. currentmodule:: ur5e_robotiq
        .. autosummary::
           :toctree: _generate
           get_ik
           get_fk
    )pbdoc";

    m.def("get_ik", [](const std::vector<std::vector<double>> &rot_list, const std::vector<double> &trans_list, const py::object &free_jt_vals = py::none())
          {
      using namespace ikfast;
      IkSolutionList<double> solutions;

      double eerot[9], eetrans[3];

      // Fill the rotation matrix
      for(std::size_t i = 0; i < 3; ++i)
      {
          std::vector<double> rot_vec = rot_list[i];
          for(std::size_t j = 0; j < 3; ++j)
          {
              eerot[3*i + j] = rot_vec[j];
          }
      }

      // Fill the translation vector
      for(std::size_t i = 0; i < 3; ++i)
      {
          eetrans[i] = trans_list[i];
      }

      // Handle optional free joint values
      std::vector<double> free_jt_vals_vec;
      if (!free_jt_vals.is_none())
      {
          free_jt_vals_vec = free_jt_vals.cast<std::vector<double>>();
          if (free_jt_vals_vec.size() != GetNumFreeParameters())
          {
              return std::vector<std::vector<double>>();
          }
      }
      else
      {
          // If no free joint values are provided, initialize with zeros
          free_jt_vals_vec = std::vector<double>(GetNumFreeParameters(), 0.0);
      }

      // Call ikfast routine
      bool b_success = ComputeIk(eetrans, eerot, free_jt_vals_vec.empty() ? nullptr : &free_jt_vals_vec[0], solutions);

      std::vector<std::vector<double>> solution_list;
      if (!b_success)
      {
          return solution_list; // Equivalent to returning None in Python
      }

      std::vector<double> solvalues(GetNumJoints());

      // Convert all ikfast solutions into a std::vector
      for(std::size_t i = 0; i < solutions.GetNumSolutions(); ++i)
      {
          const IkSolutionBase<double>& sol = solutions.GetSolution(i);
          std::vector<double> vsolfree(sol.GetFree().size());
          sol.GetSolution(&solvalues[0],
                          vsolfree.size() > 0 ? &vsolfree[0] : NULL);

          std::vector<double> individual_solution = std::vector<double>(GetNumJoints());
          for(std::size_t j = 0; j < solvalues.size(); ++j)
          {
              individual_solution[j] = solvalues[j];
          }
          solution_list.push_back(individual_solution);
      }
      return solution_list; }, py::arg("rot_list"), py::arg("trans_list"), py::arg("free_jt_vals") = py::none(),
          R"pbdoc(
        get inverse kinematic solutions for ur5e_robotiq
    )pbdoc");

    m.def("get_fk", [](const std::vector<double> &joint_list)
          {
      using namespace ikfast;
      // eerot is a flattened 3x3 rotation matrix
      double eerot[9], eetrans[3];

      std::vector<double> joints(GetNumJoints());
      for(std::size_t i = 0; i < GetNumJoints(); ++i)
      {
          joints[i] = joint_list[i];
      }

      // call ikfast routine
      ComputeFk(&joints[0], eetrans, eerot);

      // convert computed EE pose to a python object
      std::vector<double> pos(3);
      std::vector<std::vector<double>> rot(3);

      for(std::size_t i = 0; i < 3; ++i)
      {
          pos[i] = eetrans[i];
          std::vector<double> rot_vec(3);
          for( std::size_t j = 0; j < 3; ++j)
          {
              rot_vec[j] = eerot[3*i + j];
          }
          rot[i] = rot_vec;
      }
      return std::make_tuple(pos, rot); }, py::arg("joint_list"),
          R"pbdoc(
        get forward kinematic solutions for ur5e_robotiq
    )pbdoc");

    m.def("get_dof", []()
          { return int(GetNumJoints()); },
          R"pbdoc(
        get number dofs configured for the ikfast module
    )pbdoc");

    m.def("get_free_dof", []()
          { return int(GetNumFreeParameters()); },
          R"pbdoc(
        get number of free dofs configured for the ikfast module
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}