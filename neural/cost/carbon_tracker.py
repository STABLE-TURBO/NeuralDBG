"""
Carbon footprint tracking for training jobs.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from neural.cost.estimator import CloudProvider, InstanceType


@dataclass
class CarbonReport:
    """Carbon emission report."""
    
    job_id: str
    provider: CloudProvider
    region: str
    instance_type: InstanceType
    duration_hours: float
    energy_kwh: float
    carbon_kg_co2: float
    carbon_intensity: float
    equivalent_miles_driven: float
    equivalent_trees_needed: float
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'job_id': self.job_id,
            'provider': self.provider.value,
            'region': self.region,
            'instance_type': self.instance_type.name,
            'duration_hours': self.duration_hours,
            'energy_kwh': self.energy_kwh,
            'carbon_kg_co2': self.carbon_kg_co2,
            'carbon_intensity': self.carbon_intensity,
            'equivalent_miles_driven': self.equivalent_miles_driven,
            'equivalent_trees_needed': self.equivalent_trees_needed,
            'timestamp': self.timestamp,
        }


class CarbonTracker:
    """Track carbon footprint of training jobs."""
    
    CARBON_INTENSITY = {
        'us-east-1': 415.755,
        'us-west-2': 285.729,
        'eu-west-1': 316.729,
        'eu-central-1': 338.019,
        'ap-southeast-1': 392.729,
        'ap-northeast-1': 464.729,
        'us-central1': 394.449,
        'europe-west4': 308.877,
        'asia-southeast1': 431.133,
        'northeurope': 217.229,
        'westeurope': 350.877,
    }
    
    GPU_TDP_WATTS = {
        'T4': 70,
        'V100': 300,
        'A100': 400,
        'A10G': 150,
        'K80': 300,
    }
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize carbon tracker.
        
        Parameters
        ----------
        storage_path : str, optional
            Path to store carbon reports
        """
        self.storage_path = Path(storage_path) if storage_path else Path("carbon_data")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.reports: List[CarbonReport] = []
        
    def track_training(
        self,
        job_id: str,
        provider: CloudProvider,
        region: str,
        instance_type: InstanceType,
        duration_hours: float
    ) -> CarbonReport:
        """
        Track carbon emissions for training job.
        
        Parameters
        ----------
        job_id : str
            Job identifier
        provider : CloudProvider
            Cloud provider
        region : str
            Cloud region
        instance_type : InstanceType
            Instance type used
        duration_hours : float
            Training duration in hours
            
        Returns
        -------
        CarbonReport
            Carbon emission report
        """
        energy_kwh = self._calculate_energy_consumption(
            instance_type,
            duration_hours
        )
        
        carbon_intensity = self.CARBON_INTENSITY.get(region, 400.0)
        
        carbon_kg_co2 = (energy_kwh * carbon_intensity) / 1000.0
        
        equivalent_miles = carbon_kg_co2 / 0.404
        equivalent_trees = carbon_kg_co2 / 21.0
        
        report = CarbonReport(
            job_id=job_id,
            provider=provider,
            region=region,
            instance_type=instance_type,
            duration_hours=duration_hours,
            energy_kwh=energy_kwh,
            carbon_kg_co2=carbon_kg_co2,
            carbon_intensity=carbon_intensity,
            equivalent_miles_driven=equivalent_miles,
            equivalent_trees_needed=equivalent_trees
        )
        
        self.reports.append(report)
        self._save_report(report)
        
        return report
    
    def get_total_emissions(
        self,
        time_period_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get total emissions summary.
        
        Parameters
        ----------
        time_period_days : int, optional
            Time period to summarize (None for all time)
            
        Returns
        -------
        dict
            Emissions summary
        """
        reports = self.reports
        
        if time_period_days:
            cutoff_time = time.time() - (time_period_days * 86400)
            reports = [r for r in reports if r.timestamp >= cutoff_time]
        
        if not reports:
            return {
                'total_jobs': 0,
                'total_carbon_kg_co2': 0.0,
                'total_energy_kwh': 0.0,
            }
        
        total_carbon = sum(r.carbon_kg_co2 for r in reports)
        total_energy = sum(r.energy_kwh for r in reports)
        
        return {
            'total_jobs': len(reports),
            'total_carbon_kg_co2': total_carbon,
            'total_energy_kwh': total_energy,
            'equivalent_miles_driven': total_carbon / 0.404,
            'equivalent_trees_needed': total_carbon / 21.0,
            'avg_carbon_per_job': total_carbon / len(reports),
            'by_provider': self._summarize_by_field(reports, 'provider'),
            'by_region': self._summarize_by_field(reports, 'region'),
        }
    
    def compare_regions(
        self,
        instance_type: InstanceType,
        duration_hours: float
    ) -> List[Dict[str, Any]]:
        """
        Compare carbon emissions across regions.
        
        Parameters
        ----------
        instance_type : InstanceType
            Instance type
        duration_hours : float
            Training duration
            
        Returns
        -------
        list
            Comparison of regions sorted by emissions
        """
        energy_kwh = self._calculate_energy_consumption(
            instance_type,
            duration_hours
        )
        
        comparisons = []
        
        for region, intensity in self.CARBON_INTENSITY.items():
            carbon_kg = (energy_kwh * intensity) / 1000.0
            
            comparisons.append({
                'region': region,
                'carbon_kg_co2': carbon_kg,
                'carbon_intensity': intensity,
                'energy_kwh': energy_kwh,
            })
        
        return sorted(comparisons, key=lambda x: x['carbon_kg_co2'])
    
    def get_greenest_region(
        self,
        provider: CloudProvider,
        instance_type: InstanceType
    ) -> str:
        """
        Find greenest region for provider.
        
        Parameters
        ----------
        provider : CloudProvider
            Cloud provider
        instance_type : InstanceType
            Instance type
            
        Returns
        -------
        str
            Greenest region identifier
        """
        comparisons = self.compare_regions(instance_type, 1.0)
        
        provider_regions = {
            CloudProvider.AWS: ['us-east-1', 'us-west-2', 'eu-west-1', 'eu-central-1'],
            CloudProvider.GCP: ['us-central1', 'europe-west4', 'asia-southeast1'],
            CloudProvider.AZURE: ['northeurope', 'westeurope'],
        }
        
        valid_regions = provider_regions.get(provider, [])
        
        for comparison in comparisons:
            if comparison['region'] in valid_regions:
                return comparison['region']
        
        return valid_regions[0] if valid_regions else 'unknown'
    
    def estimate_offset_cost(
        self,
        carbon_kg_co2: float,
        price_per_ton: float = 15.0
    ) -> float:
        """
        Estimate carbon offset cost.
        
        Parameters
        ----------
        carbon_kg_co2 : float
            Carbon emissions in kg CO2
        price_per_ton : float
            Price per ton of CO2 offset
            
        Returns
        -------
        float
            Estimated offset cost in USD
        """
        carbon_tons = carbon_kg_co2 / 1000.0
        return carbon_tons * price_per_ton
    
    def _calculate_energy_consumption(
        self,
        instance_type: InstanceType,
        duration_hours: float
    ) -> float:
        """Calculate energy consumption in kWh."""
        cpu_tdp_watts = instance_type.cpu_count * 15
        
        gpu_tdp = 0
        if instance_type.gpu_type and instance_type.gpu_count > 0:
            for gpu_key, tdp in self.GPU_TDP_WATTS.items():
                if gpu_key in instance_type.gpu_type:
                    gpu_tdp = tdp * instance_type.gpu_count
                    break
        
        total_watts = cpu_tdp_watts + gpu_tdp
        
        pue = 1.2
        
        effective_watts = total_watts * pue
        
        kwh = (effective_watts * duration_hours) / 1000.0
        
        return kwh
    
    def _summarize_by_field(
        self,
        reports: List[CarbonReport],
        field: str
    ) -> Dict[str, float]:
        """Summarize emissions by field."""
        summary: Dict[str, float] = {}
        
        for report in reports:
            if field == 'provider':
                key = report.provider.value
            elif field == 'region':
                key = report.region
            else:
                key = 'unknown'
            
            summary[key] = summary.get(key, 0.0) + report.carbon_kg_co2
        
        return summary
    
    def _save_report(self, report: CarbonReport):
        """Save carbon report to file."""
        report_file = self.storage_path / f"carbon_{report.job_id}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
    
    def load_reports(self):
        """Load saved reports from disk."""
        for report_file in self.storage_path.glob("carbon_*.json"):
            try:
                with open(report_file, 'r') as f:
                    data = json.load(f)
                
                report = CarbonReport(
                    job_id=data['job_id'],
                    provider=CloudProvider(data['provider']),
                    region=data['region'],
                    instance_type=InstanceType(
                        name=data['instance_type'],
                        provider=CloudProvider(data['provider']),
                        cpu_count=0,
                        memory_gb=0
                    ),
                    duration_hours=data['duration_hours'],
                    energy_kwh=data['energy_kwh'],
                    carbon_kg_co2=data['carbon_kg_co2'],
                    carbon_intensity=data['carbon_intensity'],
                    equivalent_miles_driven=data['equivalent_miles_driven'],
                    equivalent_trees_needed=data['equivalent_trees_needed'],
                    timestamp=data['timestamp']
                )
                
                self.reports.append(report)
                
            except Exception:
                continue
