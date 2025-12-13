"""
Billing management and payment integration for SaaS model.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

from .models import Organization, BillingPlan


class BillingManager:
    """Manages billing and subscriptions for organizations."""
    
    def __init__(self, base_dir: str = "neural_organizations"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.billing_dir = self.base_dir / "billing"
        self.billing_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_billing_file(self, org_id: str) -> Path:
        """Get the billing data file for an organization."""
        return self.billing_dir / f"{org_id}.json"
    
    def _load_billing_data(self, org_id: str) -> Dict[str, Any]:
        """Load billing data for an organization."""
        billing_file = self._get_billing_file(org_id)
        if billing_file.exists():
            with open(billing_file, 'r') as f:
                return json.load(f)
        return {
            'org_id': org_id,
            'invoices': [],
            'payment_methods': [],
            'subscription': None,
        }
    
    def _save_billing_data(self, org_id: str, billing_data: Dict[str, Any]) -> None:
        """Save billing data for an organization."""
        billing_file = self._get_billing_file(org_id)
        with open(billing_file, 'w') as f:
            json.dump(billing_data, f, indent=2)
    
    def get_plan_pricing(self, plan: BillingPlan) -> Dict[str, Any]:
        """Get pricing information for a billing plan."""
        pricing = {
            BillingPlan.FREE: {
                'monthly_price': 0.0,
                'annual_price': 0.0,
                'currency': 'USD',
                'features': [
                    '10 models',
                    '100 experiments',
                    '10 GB storage',
                    '100 compute hours',
                    '5 team members',
                ],
            },
            BillingPlan.STARTER: {
                'monthly_price': 29.0,
                'annual_price': 290.0,
                'currency': 'USD',
                'features': [
                    '50 models',
                    '500 experiments',
                    '100 GB storage',
                    '1,000 compute hours',
                    '10 team members',
                    'Priority support',
                ],
            },
            BillingPlan.PROFESSIONAL: {
                'monthly_price': 99.0,
                'annual_price': 990.0,
                'currency': 'USD',
                'features': [
                    '200 models',
                    '2,000 experiments',
                    '500 GB storage',
                    '5,000 compute hours',
                    '50 team members',
                    'Priority support',
                    'Advanced analytics',
                ],
            },
            BillingPlan.ENTERPRISE: {
                'monthly_price': 499.0,
                'annual_price': 4990.0,
                'currency': 'USD',
                'features': [
                    'Unlimited models',
                    'Unlimited experiments',
                    'Unlimited storage',
                    'Unlimited compute hours',
                    'Unlimited team members',
                    '24/7 dedicated support',
                    'Advanced analytics',
                    'Custom integrations',
                    'SLA guarantee',
                ],
            },
        }
        return pricing.get(plan, pricing[BillingPlan.FREE])
    
    def create_subscription(
        self,
        org: Organization,
        plan: BillingPlan,
        billing_cycle: str = "monthly",
        payment_method_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a subscription for an organization."""
        billing_data = self._load_billing_data(org.org_id)
        
        pricing = self.get_plan_pricing(plan)
        amount = pricing['monthly_price'] if billing_cycle == 'monthly' else pricing['annual_price']
        
        subscription = {
            'subscription_id': f"sub_{org.org_id}_{datetime.now().timestamp()}",
            'org_id': org.org_id,
            'plan': plan.value,
            'billing_cycle': billing_cycle,
            'amount': amount,
            'currency': pricing['currency'],
            'status': 'active',
            'payment_method_id': payment_method_id,
            'start_date': datetime.now().isoformat(),
            'next_billing_date': (datetime.now() + timedelta(days=30 if billing_cycle == 'monthly' else 365)).isoformat(),
            'created_at': datetime.now().isoformat(),
        }
        
        billing_data['subscription'] = subscription
        self._save_billing_data(org.org_id, billing_data)
        
        return subscription
    
    def cancel_subscription(self, org_id: str) -> bool:
        """Cancel a subscription."""
        billing_data = self._load_billing_data(org_id)
        
        if not billing_data.get('subscription'):
            return False
        
        billing_data['subscription']['status'] = 'cancelled'
        billing_data['subscription']['cancelled_at'] = datetime.now().isoformat()
        
        self._save_billing_data(org_id, billing_data)
        
        return True
    
    def get_subscription(self, org_id: str) -> Optional[Dict[str, Any]]:
        """Get the current subscription for an organization."""
        billing_data = self._load_billing_data(org_id)
        return billing_data.get('subscription')
    
    def create_invoice(
        self,
        org_id: str,
        amount: float,
        description: str,
        items: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Create an invoice for an organization."""
        billing_data = self._load_billing_data(org_id)
        
        invoice = {
            'invoice_id': f"inv_{org_id}_{datetime.now().timestamp()}",
            'org_id': org_id,
            'amount': amount,
            'currency': 'USD',
            'description': description,
            'items': items or [],
            'status': 'pending',
            'created_at': datetime.now().isoformat(),
            'due_date': (datetime.now() + timedelta(days=30)).isoformat(),
            'paid_at': None,
        }
        
        billing_data['invoices'].append(invoice)
        self._save_billing_data(org_id, billing_data)
        
        return invoice
    
    def mark_invoice_paid(
        self,
        org_id: str,
        invoice_id: str,
        payment_method: Optional[str] = None,
    ) -> bool:
        """Mark an invoice as paid."""
        billing_data = self._load_billing_data(org_id)
        
        for invoice in billing_data['invoices']:
            if invoice['invoice_id'] == invoice_id:
                invoice['status'] = 'paid'
                invoice['paid_at'] = datetime.now().isoformat()
                if payment_method:
                    invoice['payment_method'] = payment_method
                
                self._save_billing_data(org_id, billing_data)
                return True
        
        return False
    
    def get_invoices(
        self,
        org_id: str,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get invoices for an organization."""
        billing_data = self._load_billing_data(org_id)
        invoices = billing_data.get('invoices', [])
        
        if status:
            invoices = [inv for inv in invoices if inv['status'] == status]
        
        return sorted(invoices, key=lambda x: x['created_at'], reverse=True)
    
    def add_payment_method(
        self,
        org_id: str,
        payment_method_type: str,
        last_four: str,
        is_default: bool = False,
    ) -> Dict[str, Any]:
        """Add a payment method for an organization."""
        billing_data = self._load_billing_data(org_id)
        
        if is_default:
            for pm in billing_data['payment_methods']:
                pm['is_default'] = False
        
        payment_method = {
            'payment_method_id': f"pm_{org_id}_{datetime.now().timestamp()}",
            'type': payment_method_type,
            'last_four': last_four,
            'is_default': is_default,
            'created_at': datetime.now().isoformat(),
        }
        
        billing_data['payment_methods'].append(payment_method)
        self._save_billing_data(org_id, billing_data)
        
        return payment_method
    
    def get_payment_methods(self, org_id: str) -> List[Dict[str, Any]]:
        """Get payment methods for an organization."""
        billing_data = self._load_billing_data(org_id)
        return billing_data.get('payment_methods', [])
    
    def calculate_usage_cost(
        self,
        compute_hours: float,
        storage_gb: float,
        api_calls: int,
    ) -> Dict[str, Any]:
        """Calculate cost based on usage."""
        rates = {
            'compute_hour': 0.50,
            'storage_gb_month': 0.10,
            'api_call_1000': 0.01,
        }
        
        compute_cost = compute_hours * rates['compute_hour']
        storage_cost = storage_gb * rates['storage_gb_month']
        api_cost = (api_calls / 1000) * rates['api_call_1000']
        
        total_cost = compute_cost + storage_cost + api_cost
        
        return {
            'compute_cost': round(compute_cost, 2),
            'storage_cost': round(storage_cost, 2),
            'api_cost': round(api_cost, 2),
            'total_cost': round(total_cost, 2),
            'currency': 'USD',
            'breakdown': {
                'compute': {'hours': compute_hours, 'rate': rates['compute_hour']},
                'storage': {'gb': storage_gb, 'rate': rates['storage_gb_month']},
                'api': {'calls': api_calls, 'rate': rates['api_call_1000']},
            },
        }


class StripeIntegration:
    """Integration with Stripe payment processor."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or "sk_test_dummy_key"
        self._stripe_available = False
        
        try:
            import stripe
            self.stripe = stripe
            self.stripe.api_key = self.api_key
            self._stripe_available = True
        except ImportError:
            self.stripe = None
    
    def is_available(self) -> bool:
        """Check if Stripe integration is available."""
        return self._stripe_available
    
    def create_customer(
        self,
        org: Organization,
        email: str,
    ) -> Optional[str]:
        """Create a Stripe customer."""
        if not self.is_available():
            return None
        
        try:
            customer = self.stripe.Customer.create(
                email=email,
                name=org.name,
                metadata={
                    'org_id': org.org_id,
                    'org_name': org.name,
                },
            )
            return customer.id
        except Exception:
            return None
    
    def create_subscription(
        self,
        customer_id: str,
        price_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Create a Stripe subscription."""
        if not self.is_available():
            return None
        
        try:
            subscription = self.stripe.Subscription.create(
                customer=customer_id,
                items=[{'price': price_id}],
            )
            return {
                'subscription_id': subscription.id,
                'status': subscription.status,
                'current_period_end': subscription.current_period_end,
            }
        except Exception:
            return None
    
    def cancel_subscription(self, subscription_id: str) -> bool:
        """Cancel a Stripe subscription."""
        if not self.is_available():
            return False
        
        try:
            self.stripe.Subscription.delete(subscription_id)
            return True
        except Exception:
            return False
    
    def create_payment_intent(
        self,
        amount: float,
        currency: str = "usd",
        customer_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Create a payment intent."""
        if not self.is_available():
            return None
        
        try:
            intent = self.stripe.PaymentIntent.create(
                amount=int(amount * 100),
                currency=currency,
                customer=customer_id,
            )
            return {
                'payment_intent_id': intent.id,
                'client_secret': intent.client_secret,
                'status': intent.status,
            }
        except Exception:
            return None
    
    def attach_payment_method(
        self,
        payment_method_id: str,
        customer_id: str,
    ) -> bool:
        """Attach a payment method to a customer."""
        if not self.is_available():
            return False
        
        try:
            self.stripe.PaymentMethod.attach(
                payment_method_id,
                customer=customer_id,
            )
            return True
        except Exception:
            return False
    
    def list_invoices(
        self,
        customer_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """List invoices for a customer."""
        if not self.is_available():
            return []
        
        try:
            invoices = self.stripe.Invoice.list(
                customer=customer_id,
                limit=limit,
            )
            return [
                {
                    'invoice_id': inv.id,
                    'amount': inv.amount_due / 100,
                    'currency': inv.currency,
                    'status': inv.status,
                    'created': inv.created,
                }
                for inv in invoices.data
            ]
        except Exception:
            return []
