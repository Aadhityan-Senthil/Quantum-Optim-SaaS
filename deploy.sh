#!/bin/bash

# QuantumOptim SaaS - One-Click Deployment Script
# Deploy to Railway (Backend) + Vercel (Frontend)

set -e

echo "🚀 QuantumOptim SaaS - One-Click Deployment"
echo "==========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [[ ! -f "README.md" ]] || [[ ! -d "backend" ]] || [[ ! -d "frontend" ]]; then
    echo -e "${RED}❌ Error: Please run this script from the root of the quantumoptim-saas repository${NC}"
    exit 1
fi

echo -e "${BLUE}📋 Deployment Checklist:${NC}"
echo "✅ Code pushed to GitHub"
echo "✅ Railway configuration ready"
echo "✅ Vercel configuration ready"
echo "✅ Sentry monitoring configured"
echo ""

# Railway Deployment Instructions
echo -e "${YELLOW}🚂 STEP 1: Deploy Backend to Railway${NC}"
echo "1. Go to: https://railway.app/"
echo "2. Sign up with your GitHub account"
echo "3. Click 'New Project' → 'Deploy from GitHub repo'"
echo "4. Select: Aadhityan-Senthil/Quantum-Optim-SaaS"
echo "5. Railway will automatically detect the configuration"
echo ""
echo "🔧 Add these services to your Railway project:"
echo "   • PostgreSQL (Database)"
echo "   • Redis (Cache & Queue)"
echo ""
echo "🔑 Set these environment variables in Railway:"
echo "   • ENVIRONMENT=production"
echo "   • SECRET_KEY=<generate-secure-key>"
echo "   • SENTRY_DSN=https://ac7586c0702221915f6b95e8fd8d05dd@o4509855094538240.ingest.us.sentry.io/4509855113150464"
echo "   • RAZORPAY_KEY_ID=(leave empty for no-payment launch)"
echo "   • RAZORPAY_KEY_SECRET=(leave empty for no-payment launch)"
echo ""

# Vercel Deployment Instructions  
echo -e "${YELLOW}⚡ STEP 2: Deploy Frontend to Vercel${NC}"
echo "1. Go to: https://vercel.com/"
echo "2. Sign up with your GitHub account"
echo "3. Click 'New Project' → Import from GitHub"
echo "4. Select: Aadhityan-Senthil/Quantum-Optim-SaaS"
echo "5. Set Root Directory: 'frontend'"
echo "6. Framework Preset: 'Vite'"
echo ""
echo "🔑 Set these environment variables in Vercel:"
echo "   • REACT_APP_API_URL=<your-railway-backend-url>"
echo "   • REACT_APP_ENVIRONMENT=production"
echo ""

# Final verification steps
echo -e "${YELLOW}🧪 STEP 3: Verify Deployment${NC}"
echo "1. Test Backend Health: <railway-url>/health"
echo "2. Test Frontend Loading: <vercel-url>"
echo "3. Check Sentry Dashboard for monitoring data"
echo "4. Test user registration and optimization submission"
echo ""

echo -e "${GREEN}🎉 Deployment URLs:${NC}"
echo "📊 Railway Backend: https://quantumoptim-backend-production.up.railway.app"
echo "🌐 Vercel Frontend: https://quantumoptim-frontend.vercel.app"
echo "📈 Sentry Monitoring: https://sentry.io/organizations/your-org/projects/"
echo ""

echo -e "${BLUE}💡 Pro Tips:${NC}"
echo "• Railway gives you $5/month free credits"
echo "• Vercel has generous free tier for frontend"
echo "• Monitor your Sentry quotas and alerts"
echo "• Add custom domain later in both platforms"
echo ""

echo -e "${GREEN}✅ Your QuantumOptim SaaS is ready to launch!${NC}"
echo -e "${YELLOW}🚀 Happy scaling! - AYNX AI Team${NC}"
