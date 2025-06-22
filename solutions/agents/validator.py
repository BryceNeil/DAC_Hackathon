"""
Validator Agent
===============

Validates the quality and completeness of generated designs and results.
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from tools.file_manager import FileManager


class Validator:
    """Agent responsible for validating design quality and completeness"""
    
    def __init__(self):
        self.name = "Validator"
        self.file_manager = FileManager()
        
        # Quality thresholds
        self.timing_thresholds = {
            'max_wns': -0.1,  # Worst Negative Slack threshold (ns)
            'max_tns': -1.0,  # Total Negative Slack threshold (ns)
            'min_hold_slack': 0.0  # Minimum hold slack (ns)
        }
        
        self.area_thresholds = {
            'max_utilization': 80.0,  # Maximum core utilization (%)
            'max_congestion': 90.0   # Maximum routing congestion (%)
        }
    
    def validate_complete_flow(self, output_dir: str, analysis: Dict[str, Any], 
                              verification_results: Dict[str, Any],
                              physical_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the complete tapeout flow results
        
        Args:
            output_dir: Directory containing generated files
            analysis: Analyzed specification data
            verification_results: Verification results
            physical_results: Physical design results
            
        Returns:
            Dict containing validation results and overall score
        """
        validation_results = {
            'overall_pass': False,
            'overall_score': 0.0,  # 0-100 scale
            'file_validation': {},
            'rtl_validation': {},
            'verification_validation': {},
            'physical_validation': {},
            'timing_validation': {},
            'recommendations': [],
            'critical_issues': [],
            'warnings': []
        }
        
        try:
            # Validate required files
            file_results = self._validate_required_files(output_dir, analysis['problem_name'])
            validation_results['file_validation'] = file_results
            
            # Validate RTL quality
            rtl_file = os.path.join(output_dir, f"{analysis['problem_name']}.v")
            if os.path.exists(rtl_file):
                rtl_results = self._validate_rtl_quality(rtl_file, analysis)
                validation_results['rtl_validation'] = rtl_results
            
            # Validate verification results
            verification_validation = self._validate_verification_results(verification_results)
            validation_results['verification_validation'] = verification_validation
            
            # Validate physical design results
            physical_validation = self._validate_physical_results(physical_results)
            validation_results['physical_validation'] = physical_validation
            
            # Validate timing closure
            timing_validation = self._validate_timing_closure(physical_results.get('timing_results', {}))
            validation_results['timing_validation'] = timing_validation
            
            # Calculate overall score and pass/fail
            overall_score, overall_pass = self._calculate_overall_score(validation_results)
            validation_results['overall_score'] = overall_score
            validation_results['overall_pass'] = overall_pass
            
            # Generate recommendations
            recommendations = self._generate_recommendations(validation_results)
            validation_results['recommendations'] = recommendations
        
        except Exception as e:
            validation_results['critical_issues'].append(f"Validation error: {str(e)}")
        
        return validation_results
    
    def _validate_required_files(self, output_dir: str, problem_name: str) -> Dict[str, Any]:
        """Validate presence and basic properties of required files"""
        
        required_files = {
            f"{problem_name}.v": "RTL file",
            "6_final.sdc": "SDC constraints file", 
            "6_final.odb": "Final ODB database"
        }
        
        results = {
            'files_present': {},
            'files_valid': {},
            'missing_files': [],
            'invalid_files': []
        }
        
        for filename, description in required_files.items():
            filepath = os.path.join(output_dir, filename)
            
            # Check file presence
            if os.path.exists(filepath):
                results['files_present'][filename] = True
                
                # Check file validity
                valid = self._check_file_validity(filepath, filename)
                results['files_valid'][filename] = valid
                
                if not valid:
                    results['invalid_files'].append(f"{filename} ({description})")
            else:
                results['files_present'][filename] = False
                results['missing_files'].append(f"{filename} ({description})")
        
        return results
    
    def _check_file_validity(self, filepath: str, filename: str) -> bool:
        """Check if a file has valid content"""
        
        try:
            if not os.path.exists(filepath):
                return False
            
            # Check file size (not empty)
            if os.path.getsize(filepath) == 0:
                return False
            
            with open(filepath, 'r') as f:
                content = f.read().strip()
            
            # File-specific validation
            if filename.endswith('.v') or filename.endswith('.sv'):
                # RTL file validation
                return 'module ' in content and 'endmodule' in content
            elif filename.endswith('.sdc'):
                # SDC file validation
                return any(cmd in content for cmd in ['create_clock', 'set_input_delay', 'set_output_delay'])
            elif filename.endswith('.odb'):
                # ODB file validation (basic check)
                return len(content) > 10  # Should have some content
            
            return True
            
        except Exception:
            return False
    
    def _validate_rtl_quality(self, rtl_file: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Validate RTL code quality"""
        
        results = {
            'syntax_valid': False,
            'has_reset': False,
            'has_clock': False,
            'proper_naming': False,
            'code_quality_score': 0.0,
            'issues': []
        }
        
        try:
            with open(rtl_file, 'r') as f:
                content = f.read()
            
            # Basic syntax validation
            if 'module ' in content and 'endmodule' in content:
                results['syntax_valid'] = True
            else:
                results['issues'].append("Invalid module structure")
            
            # Check for clock and reset
            if any(signal in content.lower() for signal in ['clk', 'clock']):
                results['has_clock'] = True
            else:
                results['issues'].append("No clock signal detected")
            
            if any(signal in content.lower() for signal in ['rst', 'reset']):
                results['has_reset'] = True
            else:
                results['issues'].append("No reset signal detected")
            
            # Check proper naming
            expected_name = analysis['problem_name']
            if f"module {expected_name}" in content:
                results['proper_naming'] = True
            else:
                results['issues'].append(f"Module name doesn't match expected: {expected_name}")
            
            # Calculate quality score
            quality_score = 0
            if results['syntax_valid']:
                quality_score += 40
            if results['has_clock']:
                quality_score += 20
            if results['has_reset']:
                quality_score += 20
            if results['proper_naming']:
                quality_score += 10
            
            # Additional quality checks
            if 'always_ff' in content or 'always @(posedge' in content:
                quality_score += 10  # Sequential logic
            
            results['code_quality_score'] = quality_score
        
        except Exception as e:
            results['issues'].append(f"RTL validation error: {str(e)}")
        
        return results
    
    def _validate_verification_results(self, verification_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate verification results"""
        
        results = {
            'compilation_passed': False,
            'simulation_passed': False,
            'testbench_found': False,
            'verification_score': 0.0,
            'issues': []
        }
        
        try:
            results['compilation_passed'] = verification_results.get('compilation_success', False)
            results['simulation_passed'] = verification_results.get('simulation_success', False)
            results['testbench_found'] = verification_results.get('testbench_found', False)
            
            # Calculate verification score
            score = 0
            if results['testbench_found']:
                score += 30
            if results['compilation_passed']:
                score += 40
            if results['simulation_passed']:
                score += 30
            
            results['verification_score'] = score
            
            # Collect issues
            if verification_results.get('errors'):
                results['issues'].extend(verification_results['errors'])
        
        except Exception as e:
            results['issues'].append(f"Verification validation error: {str(e)}")
        
        return results
    
    def _validate_physical_results(self, physical_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate physical design results"""
        
        results = {
            'synthesis_passed': False,
            'placement_passed': False,
            'routing_passed': False,
            'final_odb_exists': False,
            'physical_score': 0.0,
            'issues': []
        }
        
        try:
            results['synthesis_passed'] = physical_results.get('synthesis_success', False)
            results['placement_passed'] = physical_results.get('placement_success', False) 
            results['routing_passed'] = physical_results.get('routing_success', False)
            
            # Check ODB file
            odb_path = physical_results.get('final_odb_path', '')
            if odb_path and os.path.exists(odb_path):
                results['final_odb_exists'] = True
            
            # Calculate physical score
            score = 0
            if results['synthesis_passed']:
                score += 25
            if results['placement_passed']:
                score += 25
            if results['routing_passed']:
                score += 25
            if results['final_odb_exists']:
                score += 25
            
            results['physical_score'] = score
            
            # Collect issues
            if physical_results.get('errors'):
                results['issues'].extend(physical_results['errors'])
        
        except Exception as e:
            results['issues'].append(f"Physical validation error: {str(e)}")
        
        return results
    
    def _validate_timing_closure(self, timing_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate timing closure"""
        
        results = {
            'setup_timing_met': False,
            'hold_timing_met': False,
            'timing_score': 0.0,
            'wns': None,
            'tns': None,
            'hold_slack': None,
            'issues': []
        }
        
        try:
            wns = timing_results.get('worst_negative_slack', 0.0)
            tns = timing_results.get('total_negative_slack', 0.0)
            hold_slack = timing_results.get('worst_hold_slack', 0.0)
            
            results['wns'] = wns
            results['tns'] = tns
            results['hold_slack'] = hold_slack
            
            # Check setup timing
            if wns >= self.timing_thresholds['max_wns']:
                results['setup_timing_met'] = True
            else:
                results['issues'].append(f"Setup timing violation: WNS = {wns}ns")
            
            # Check hold timing
            if hold_slack >= self.timing_thresholds['min_hold_slack']:
                results['hold_timing_met'] = True
            else:
                results['issues'].append(f"Hold timing violation: Slack = {hold_slack}ns")
            
            # Calculate timing score
            score = 0
            if results['setup_timing_met']:
                score += 50
            if results['hold_timing_met']:
                score += 50
            
            results['timing_score'] = score
        
        except Exception as e:
            results['issues'].append(f"Timing validation error: {str(e)}")
        
        return results
    
    def _calculate_overall_score(self, validation_results: Dict[str, Any]) -> Tuple[float, bool]:
        """Calculate overall validation score and pass/fail status"""
        
        # Weight different validation categories
        weights = {
            'file_validation': 0.2,
            'rtl_validation': 0.25,
            'verification_validation': 0.25,
            'physical_validation': 0.2,
            'timing_validation': 0.1
        }
        
        total_score = 0.0
        
        # File validation score
        file_results = validation_results['file_validation']
        file_score = (len(file_results.get('files_present', {})) - len(file_results.get('missing_files', []))) / 3 * 100
        total_score += file_score * weights['file_validation']
        
        # RTL validation score
        rtl_score = validation_results['rtl_validation'].get('code_quality_score', 0.0)
        total_score += rtl_score * weights['rtl_validation']
        
        # Verification validation score
        verification_score = validation_results['verification_validation'].get('verification_score', 0.0)
        total_score += verification_score * weights['verification_validation']
        
        # Physical validation score
        physical_score = validation_results['physical_validation'].get('physical_score', 0.0)
        total_score += physical_score * weights['physical_validation']
        
        # Timing validation score
        timing_score = validation_results['timing_validation'].get('timing_score', 0.0)
        total_score += timing_score * weights['timing_validation']
        
        # Overall pass threshold
        pass_threshold = 70.0
        overall_pass = total_score >= pass_threshold
        
        return total_score, overall_pass
    
    def _generate_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improving the design"""
        
        recommendations = []
        
        # File-related recommendations
        file_results = validation_results['file_validation']
        if file_results.get('missing_files'):
            recommendations.append("Complete the missing output files for full tapeout readiness")
        
        # RTL-related recommendations
        rtl_results = validation_results['rtl_validation']
        if not rtl_results.get('has_reset', True):
            recommendations.append("Add proper reset handling to RTL for better robustness")
        if rtl_results.get('code_quality_score', 0) < 80:
            recommendations.append("Improve RTL code quality with better structure and naming")
        
        # Verification-related recommendations
        verification_results = validation_results['verification_validation']
        if not verification_results.get('simulation_passed', True):
            recommendations.append("Fix verification issues to ensure functional correctness")
        
        # Physical design recommendations
        physical_results = validation_results['physical_validation']
        if not physical_results.get('routing_passed', True):
            recommendations.append("Address routing congestion issues for better manufacturability")
        
        # Timing recommendations
        timing_results = validation_results['timing_validation']
        if not timing_results.get('setup_timing_met', True):
            recommendations.append("Optimize design for better timing closure")
        
        # Overall score recommendations
        if validation_results['overall_score'] < 80:
            recommendations.append("Consider design iteration to improve overall quality metrics")
        
        return recommendations 