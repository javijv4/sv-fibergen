#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""Test script for the LaplaceSolver class.

This script tests the LaplaceSolver class by running the Laplace solver
on example meshes using both Bayer and Doste methods, and compares
the result_001.vtu outputs with reference results in example_og/.
"""

import os
import argparse
import numpy as np
import pyvista as pv
from src.LaplaceSolver import LaplaceSolver


# Fields expected in each method's result_001.vtu
BAYER_FIELDS = ['Phi_BiV_EPI', 'Phi_BiV_LV', 'Phi_BiV_RV', 'Phi_BiV_AB']
DOSTE_FIELDS = ['Trans_BiV', 'Long_AV', 'Long_MV', 'Long_PV', 'Long_TV', 
                'Weight_LV', 'Weight_RV', 'Trans_EPI', 'Trans_LV', 'Trans_RV']


def compare_meshes(test_mesh, ref_mesh, fields_to_compare, tolerance=1e-6):
    """Compare two meshes and report differences in specified fields.
    
    Args:
        test_mesh: PyVista mesh from the test run.
        ref_mesh: PyVista mesh from the reference output.
        fields_to_compare: List of field names to compare.
        tolerance: Maximum allowed difference (default: 1e-6).
        
    Returns:
        dict: Comparison results for each field.
    """
    results = {}
    
    for field in fields_to_compare:
        if field not in test_mesh.point_data:
            results[field] = {"status": "MISSING", "message": f"Field '{field}' not in test mesh"}
            continue
        if field not in ref_mesh.point_data:
            results[field] = {"status": "MISSING", "message": f"Field '{field}' not in reference mesh"}
            continue
        
        test_data = np.asarray(test_mesh.point_data[field])
        ref_data = np.asarray(ref_mesh.point_data[field])
        
        if test_data.shape != ref_data.shape:
            results[field] = {
                "status": "SHAPE_MISMATCH",
                "message": f"Shape mismatch: test {test_data.shape} vs ref {ref_data.shape}"
            }
            continue
        
        # Compute differences
        diff = np.abs(test_data - ref_data)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        if max_diff <= tolerance:
            results[field] = {
                "status": "PASS",
                "max_diff": max_diff,
                "mean_diff": mean_diff
            }
        else:
            results[field] = {
                "status": "FAIL",
                "max_diff": max_diff,
                "mean_diff": mean_diff,
                "message": f"Max diff {max_diff:.2e} exceeds tolerance {tolerance:.2e}"
            }
    
    return results


def print_comparison_results(results, method_name):
    """Print comparison results in a formatted way."""
    print(f"\n{'='*60}")
    print(f"Comparison Results: {method_name}")
    print('='*60)
    
    all_passed = True
    for field, result in results.items():
        status = result["status"]
        if status == "PASS":
            print(f"  ✓ {field}: PASS (max_diff={result['max_diff']:.2e}, mean_diff={result['mean_diff']:.2e})")
        elif status == "FAIL":
            print(f"  ✗ {field}: FAIL - {result['message']}")
            all_passed = False
        else:
            print(f"  ? {field}: {status} - {result['message']}")
            all_passed = False
    
    print('-'*60)
    if all_passed:
        print(f"  Overall: ALL TESTS PASSED")
    else:
        print(f"  Overall: SOME TESTS FAILED")
    print('='*60)
    
    return all_passed


def test_bayer_method(run_solver=True):
    """Test LaplaceSolver with the Bayer method."""
    print("\n" + "="*60)
    print("Testing LaplaceSolver with BAYER method")
    print("="*60 + "\n")
    
    # Paths - use example_og for input data
    mesh_path = os.path.abspath("example_og/truncated/VOLUME.vtu")
    surfaces_dir = os.path.join(os.path.dirname(mesh_path), "mesh-surfaces")
    outdir = os.path.abspath("example_og/truncated/output_bayer_test")
    ref_dir = os.path.abspath("example_og/truncated/output_b")
    
    # Surface paths (full paths)
    surface_paths = {
        'epi': os.path.join(surfaces_dir, 'EPI.vtp'),
        'epi_apex': os.path.join(surfaces_dir, 'EPI_APEX.vtp'),
        'base': os.path.join(surfaces_dir, 'BASE.vtp'),
        'endo_lv': os.path.join(surfaces_dir, 'LV.vtp'),
        'endo_rv': os.path.join(surfaces_dir, 'RV.vtp'),
    }
    
    svfsi_exec = "svmultiphysics "
    
    # Create LaplaceSolver
    solver = LaplaceSolver(
        mesh_path=mesh_path,
        surface_paths=surface_paths,
        exec_svmultiphysics=svfsi_exec
    )
    
    print(f"Mesh path: {solver.mesh_path}")
    print(f"Output directory: {outdir}")
    print(f"Reference directory: {ref_dir}")
    
    # Run the solver
    if run_solver:
        laplace_results_file = solver.run("bayer", outdir)
    else:
        laplace_results_file = os.path.join(outdir, 'result_001.vtu')
        print(f"Skipping solver, expecting results at: {laplace_results_file}")
    
    # Compare with reference output
    ref_results_file = os.path.join(ref_dir, "result_001.vtu")
    
    if not os.path.exists(laplace_results_file):
        print(f"\nERROR: Test results not found: {laplace_results_file}")
        return None, False
        
    if not os.path.exists(ref_results_file):
        print(f"\nERROR: Reference results not found: {ref_results_file}")
        return None, False
    
    print(f"\nComparing Laplace solutions:")
    print(f"  Test: {laplace_results_file}")
    print(f"  Ref:  {ref_results_file}")
    
    test_mesh = pv.read(laplace_results_file)
    ref_mesh = pv.read(ref_results_file)
    
    comparison = compare_meshes(test_mesh, ref_mesh, BAYER_FIELDS)
    passed = print_comparison_results(comparison, "Bayer Method - Laplace Solutions")
    
    return test_mesh, passed


def test_doste_method(run_solver=True):
    """Test LaplaceSolver with the Doste method."""
    print("\n" + "="*60)
    print("Testing LaplaceSolver with DOSTE method")
    print("="*60 + "\n")
    
    # Paths - use example_og for input data
    mesh_path = os.path.abspath("example_og/ot/mesh-complete.mesh.vtu")
    surfaces_dir = os.path.join(os.path.dirname(mesh_path), "mesh-surfaces")
    outdir = os.path.abspath("example_og/ot/output_doste_test")
    ref_dir = os.path.abspath("example_og/ot/output_d")
    
    # Surface paths (full paths)
    surface_paths = {
        'epi': os.path.join(surfaces_dir, 'epi.vtp'),
        'epi_apex': os.path.join(surfaces_dir, 'epi_apex.vtp'),
        'av': os.path.join(surfaces_dir, 'av.vtp'),
        'mv': os.path.join(surfaces_dir, 'mv.vtp'),
        'tv': os.path.join(surfaces_dir, 'tv.vtp'),
        'pv': os.path.join(surfaces_dir, 'pv.vtp'),
        'base': os.path.join(surfaces_dir, 'top.vtp'),
        'endo_lv': os.path.join(surfaces_dir, 'endo_lv.vtp'),
        'endo_rv': os.path.join(surfaces_dir, 'endo_rv.vtp'),
    }
    
    svfsi_exec = "svmultiphysics "
    
    # Create LaplaceSolver
    solver = LaplaceSolver(
        mesh_path=mesh_path,
        surface_paths=surface_paths,
        exec_svmultiphysics=svfsi_exec
    )
    
    print(f"Mesh path: {solver.mesh_path}")
    print(f"Output directory: {outdir}")
    print(f"Reference directory: {ref_dir}")
    
    # Run the solver
    if run_solver:
        laplace_results_file = solver.run("doste", outdir)
    else:
        laplace_results_file = os.path.join(outdir, 'result_001.vtu')
        print(f"Skipping solver, expecting results at: {laplace_results_file}")
    
    # Compare with reference output
    ref_results_file = os.path.join(ref_dir, "result_001.vtu")
    
    if not os.path.exists(laplace_results_file):
        print(f"\nERROR: Test results not found: {laplace_results_file}")
        return None, False
        
    if not os.path.exists(ref_results_file):
        print(f"\nERROR: Reference results not found: {ref_results_file}")
        return None, False
    
    print(f"\nComparing Laplace solutions:")
    print(f"  Test: {laplace_results_file}")
    print(f"  Ref:  {ref_results_file}")
    
    test_mesh = pv.read(laplace_results_file)
    ref_mesh = pv.read(ref_results_file)
    
    comparison = compare_meshes(test_mesh, ref_mesh, DOSTE_FIELDS)
    passed = print_comparison_results(comparison, "Doste Method - Laplace Solutions")
    
    return test_mesh, passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test LaplaceSolver class")
    parser.add_argument(
        "--method", 
        choices=["bayer", "doste", "both"],
        default="both",
        help="Which method to test (default: both)"
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Skip running the solver, compare existing results"
    )
    args = parser.parse_args()
    
    run_solver = not args.no_run
    results = {}
    
    if args.method in ["bayer", "both"]:
        _, passed = test_bayer_method(run_solver)
        results["bayer"] = passed
        
    if args.method in ["doste", "both"]:
        _, passed = test_doste_method(run_solver)
        results["doste"] = passed
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    all_passed = True
    for method, passed in results.items():
        if passed is None:
            print(f"  {method.upper()}: ERROR - Could not compare")
            all_passed = False
        elif passed:
            print(f"  {method.upper()}: ✓ PASSED")
        else:
            print(f"  {method.upper()}: ✗ FAILED")
            all_passed = False
    
    print("="*60)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("="*60)
