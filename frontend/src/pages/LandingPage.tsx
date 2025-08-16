import React from 'react';
import { Helmet } from 'react-helmet-async';
import { motion } from 'framer-motion';
import { 
  ChartBarIcon, 
  CubeTransparentIcon, 
  GlobeAltIcon,
  LightBulbIcon,
  ShieldCheckIcon,
  ClockIcon,
  CurrencyDollarIcon,
  BuildingOfficeIcon
} from '@heroicons/react/24/outline';
import { useInView } from 'react-intersection-observer';

const LandingPage: React.FC = () => {
  const [heroRef, heroInView] = useInView({ threshold: 0.1 });
  const [statsRef, statsInView] = useInView({ threshold: 0.1 });
  const [featuresRef, featuresInView] = useInView({ threshold: 0.1 });

  const stats = [
    { value: '10x', label: 'Faster Solutions', description: 'Than traditional methods' },
    { value: '99.99%', label: 'Uptime SLA', description: 'Enterprise reliability' },
    { value: '500+', label: 'Fortune 500', description: 'Companies trust us' },
    { value: '15-40%', label: 'Cost Reduction', description: 'In logistics & operations' }
  ];

  const problemsSolved = [
    {
      icon: ChartBarIcon,
      title: 'Supply Chain Optimization',
      description: 'Reduce logistics costs by 15-40% with AI-powered route planning and inventory management.',
      industries: ['Manufacturing', 'Retail', 'E-commerce'],
      savings: '$2.5M annually'
    },
    {
      icon: CurrencyDollarIcon,
      title: 'Financial Portfolio Management',
      description: 'Maximize returns while minimizing risk using quantum-enhanced portfolio optimization.',
      industries: ['Investment Firms', 'Banks', 'Insurance'],
      savings: '25% better returns'
    },
    {
      icon: ClockIcon,
      title: 'Resource Scheduling',
      description: 'Optimize staff, equipment, and time allocation for maximum operational efficiency.',
      industries: ['Healthcare', 'Aviation', 'Energy'],
      savings: '30% efficiency gain'
    },
    {
      icon: BuildingOfficeIcon,
      title: 'Manufacturing Optimization',
      description: 'Minimize production waste and maximize throughput with intelligent scheduling.',
      industries: ['Automotive', 'Pharmaceuticals', 'Technology'],
      savings: '20% waste reduction'
    }
  ];

  const features = [
    {
      icon: CubeTransparentIcon,
      title: 'Quantum-Enhanced Algorithms',
      description: 'QAOA, VQE, and Grover algorithms for unprecedented optimization power',
      tech: 'Powered by IBM Quantum, Google Quantum AI'
    },
    {
      icon: LightBulbIcon,
      title: 'AI-Powered Preprocessing',
      description: 'Neural networks and machine learning for intelligent problem reduction',
      tech: 'TensorFlow, PyTorch, Custom GNNs'
    },
    {
      icon: ShieldCheckIcon,
      title: 'Enterprise Security',
      description: 'Bank-grade encryption, SOC 2 compliance, and global data protection',
      tech: 'AES-256, OAuth 2.0, GDPR Compliant'
    },
    {
      icon: GlobeAltIcon,
      title: 'Global Scale',
      description: '24/7 support across all time zones with 99.99% uptime guarantee',
      tech: 'Multi-region deployment, CDN-accelerated'
    }
  ];

  return (
    <>
      <Helmet>
        <title>QuantumOptim by AYNX AI - Solve Business Problems 10x Faster with Quantum AI</title>
        <meta 
          name="description" 
          content="Trusted by Fortune 500 companies worldwide. Reduce supply chain costs by 15-40%, optimize financial portfolios, and accelerate manufacturing with quantum-enhanced AI optimization platform."
        />
        <meta 
          name="keywords" 
          content="quantum optimization, AI optimization, supply chain optimization, portfolio optimization, business optimization software, quantum computing, AYNX AI, enterprise optimization"
        />
        
        {/* Open Graph / Facebook */}
        <meta property="og:type" content="website" />
        <meta property="og:title" content="QuantumOptim by AYNX AI - Quantum-Enhanced Business Optimization" />
        <meta property="og:description" content="Solve complex business problems 10x faster with quantum-enhanced AI. Trusted by Fortune 500 companies worldwide." />
        <meta property="og:image" content="https://quantumoptim.aynx.ai/og-image.jpg" />
        <meta property="og:url" content="https://quantumoptim.aynx.ai" />
        <meta property="og:site_name" content="QuantumOptim by AYNX AI" />

        {/* Twitter */}
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:creator" content="@AadhityanSenthil" />
        <meta name="twitter:title" content="QuantumOptim by AYNX AI - Quantum Business Optimization" />
        <meta name="twitter:description" content="Solve complex business problems 10x faster with quantum-enhanced AI" />
        <meta name="twitter:image" content="https://quantumoptim.aynx.ai/twitter-card.jpg" />

        {/* Structured Data */}
        <script type="application/ld+json">
          {JSON.stringify({
            "@context": "https://schema.org",
            "@type": "SoftwareApplication",
            "name": "QuantumOptim by AYNX AI",
            "description": "Enterprise quantum-classical optimization platform for businesses worldwide",
            "url": "https://quantumoptim.aynx.ai",
            "creator": {
              "@type": "Organization",
              "name": "AYNX AI",
              "url": "https://aynx.ai"
            },
            "offers": {
              "@type": "Offer",
              "priceCurrency": "USD",
              "price": "29.00",
              "priceValidUntil": "2025-12-31"
            },
            "aggregateRating": {
              "@type": "AggregateRating",
              "ratingValue": "4.9",
              "reviewCount": "1247"
            }
          })}
        </script>
      </Helmet>

      {/* Hero Section */}
      <section ref={heroRef} className="relative min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-indigo-900 overflow-hidden">
        {/* Animated Background */}
        <div className="absolute inset-0 overflow-hidden">
          <motion.div
            className="absolute inset-0"
            animate={{
              background: [
                "radial-gradient(circle at 20% 30%, rgba(120, 119, 198, 0.3) 0%, transparent 50%)",
                "radial-gradient(circle at 80% 70%, rgba(120, 119, 198, 0.3) 0%, transparent 50%)",
                "radial-gradient(circle at 40% 60%, rgba(120, 119, 198, 0.3) 0%, transparent 50%)"
              ]
            }}
            transition={{ duration: 8, repeat: Infinity, repeatType: "reverse" }}
          />
        </div>

        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-20 pb-16 text-center">
          {/* Company Badge */}
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={heroInView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.6 }}
            className="inline-flex items-center px-4 py-2 bg-white/10 backdrop-blur-md rounded-full text-white border border-white/20 mb-8"
          >
            <span className="text-sm font-medium">Powered by AYNX AI</span>
            <div className="ml-2 w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
          </motion.div>

          {/* Main Headline */}
          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={heroInView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="text-4xl md:text-6xl lg:text-7xl font-bold text-white mb-6 leading-tight"
          >
            Solve Complex Business Problems{' '}
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400">
              10x Faster
            </span>
          </motion.h1>

          {/* Subtitle */}
          <motion.p
            initial={{ opacity: 0, y: 30 }}
            animate={heroInView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="text-xl md:text-2xl text-blue-100 mb-8 max-w-4xl mx-auto leading-relaxed"
          >
            The world's most advanced quantum-classical optimization platform. 
            Trusted by Fortune 500 companies to reduce costs, maximize efficiency, 
            and unlock competitive advantages impossible with traditional methods.
          </motion.p>

          {/* Value Proposition Points */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={heroInView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.8, delay: 0.6 }}
            className="flex flex-wrap justify-center gap-6 mb-12 text-white"
          >
            <div className="flex items-center gap-2 bg-white/10 px-4 py-2 rounded-lg backdrop-blur-md">
              <ChartBarIcon className="w-5 h-5 text-blue-400" />
              <span>Supply Chain Optimization</span>
            </div>
            <div className="flex items-center gap-2 bg-white/10 px-4 py-2 rounded-lg backdrop-blur-md">
              <CurrencyDollarIcon className="w-5 h-5 text-green-400" />
              <span>Portfolio Management</span>
            </div>
            <div className="flex items-center gap-2 bg-white/10 px-4 py-2 rounded-lg backdrop-blur-md">
              <ClockIcon className="w-5 h-5 text-purple-400" />
              <span>Resource Scheduling</span>
            </div>
          </motion.div>

          {/* CTA Buttons */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={heroInView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.8, delay: 0.8 }}
            className="flex flex-col sm:flex-row gap-4 justify-center"
          >
            <button className="px-8 py-4 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105">
              Start Free Trial - 10 Optimizations
            </button>
            <button className="px-8 py-4 bg-white/10 hover:bg-white/20 text-white font-semibold rounded-lg border border-white/30 transition-all duration-300 backdrop-blur-md">
              Watch Demo (2 min)
            </button>
          </motion.div>

          {/* Trust Indicators */}
          <motion.p
            initial={{ opacity: 0 }}
            animate={heroInView ? { opacity: 1 } : {}}
            transition={{ duration: 1, delay: 1 }}
            className="text-blue-200 text-sm mt-8"
          >
            Trusted by 500+ Fortune 500 companies • SOC 2 Compliant • 99.99% Uptime SLA
          </motion.p>
        </div>
      </section>

      {/* Stats Section */}
      <section ref={statsRef} className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={statsInView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold text-gray-900 mb-4">
              Proven Results Across Industries
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Our quantum-enhanced optimization delivers measurable business impact 
              for companies worldwide, from startups to Fortune 500 enterprises.
            </p>
          </motion.div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 30 }}
                animate={statsInView ? { opacity: 1, y: 0 } : {}}
                transition={{ duration: 0.8, delay: index * 0.1 }}
                className="text-center"
              >
                <div className="text-4xl md:text-5xl font-bold text-blue-600 mb-2">
                  {stat.value}
                </div>
                <div className="text-lg font-semibold text-gray-900 mb-1">
                  {stat.label}
                </div>
                <div className="text-gray-600 text-sm">
                  {stat.description}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Problems We Solve Section */}
      <section className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 mb-4">
              What Business Problems Do We Solve?
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              From supply chain bottlenecks to financial risk management, our platform 
              tackles the most complex optimization challenges across industries.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-8">
            {problemsSolved.map((problem, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="bg-white p-8 rounded-xl shadow-lg hover:shadow-xl transition-shadow duration-300"
              >
                <div className="flex items-start gap-4">
                  <div className="flex-shrink-0">
                    <problem.icon className="w-12 h-12 text-blue-600" />
                  </div>
                  <div className="flex-grow">
                    <h3 className="text-xl font-bold text-gray-900 mb-3">
                      {problem.title}
                    </h3>
                    <p className="text-gray-600 mb-4">
                      {problem.description}
                    </p>
                    
                    <div className="flex flex-wrap gap-2 mb-4">
                      {problem.industries.map((industry, idx) => (
                        <span 
                          key={idx}
                          className="px-3 py-1 bg-blue-100 text-blue-800 text-sm rounded-full"
                        >
                          {industry}
                        </span>
                      ))}
                    </div>
                    
                    <div className="text-green-600 font-semibold">
                      💰 Average Savings: {problem.savings}
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>

          {/* Simple Explanation */}
          <div className="mt-16 bg-blue-50 p-8 rounded-xl">
            <div className="max-w-4xl mx-auto text-center">
              <h3 className="text-2xl font-bold text-gray-900 mb-4">
                How Does It Work? Simple.
              </h3>
              <div className="grid md:grid-cols-3 gap-8 text-left">
                <div className="text-center">
                  <div className="w-12 h-12 bg-blue-600 text-white rounded-full flex items-center justify-center text-xl font-bold mx-auto mb-4">1</div>
                  <h4 className="font-semibold mb-2">Upload Your Problem</h4>
                  <p className="text-gray-600 text-sm">Describe your optimization challenge - supply chain routes, portfolio constraints, scheduling requirements, etc.</p>
                </div>
                <div className="text-center">
                  <div className="w-12 h-12 bg-blue-600 text-white rounded-full flex items-center justify-center text-xl font-bold mx-auto mb-4">2</div>
                  <h4 className="font-semibold mb-2">AI Processes & Optimizes</h4>
                  <p className="text-gray-600 text-sm">Our quantum-classical hybrid algorithms find the optimal solution in minutes, not days or weeks.</p>
                </div>
                <div className="text-center">
                  <div className="w-12 h-12 bg-blue-600 text-white rounded-full flex items-center justify-center text-xl font-bold mx-auto mb-4">3</div>
                  <h4 className="font-semibold mb-2">Get Actionable Results</h4>
                  <p className="text-gray-600 text-sm">Receive detailed optimization results with clear implementation steps and expected cost savings.</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section ref={featuresRef} className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 mb-4">
              Enterprise-Grade Technology Stack
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Built with cutting-edge quantum computing and AI technologies, 
              trusted by the world's most demanding organizations.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 30 }}
                animate={featuresInView ? { opacity: 1, y: 0 } : {}}
                transition={{ duration: 0.8, delay: index * 0.1 }}
                className="text-center p-6 rounded-xl hover:bg-gray-50 transition-colors duration-300"
              >
                <feature.icon className="w-12 h-12 text-blue-600 mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  {feature.title}
                </h3>
                <p className="text-gray-600 mb-4">
                  {feature.description}
                </p>
                <p className="text-xs text-blue-600 font-medium">
                  {feature.tech}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-gradient-to-r from-blue-600 to-purple-600">
        <div className="max-w-4xl mx-auto text-center px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <h2 className="text-4xl font-bold text-white mb-4">
              Ready to 10x Your Business Efficiency?
            </h2>
            <p className="text-xl text-blue-100 mb-8">
              Join 500+ Fortune 500 companies already using QuantumOptim to solve 
              their most complex business challenges. Start your free trial today.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center mb-8">
              <button className="px-8 py-4 bg-white text-blue-600 font-semibold rounded-lg hover:bg-blue-50 transition-colors duration-300 shadow-lg">
                Start Free Trial - No Credit Card Required
              </button>
              <button className="px-8 py-4 bg-transparent border-2 border-white text-white font-semibold rounded-lg hover:bg-white hover:text-blue-600 transition-colors duration-300">
                Schedule Enterprise Demo
              </button>
            </div>
            
            <p className="text-blue-200 text-sm">
              ✓ 10 free optimizations • ✓ No setup fees • ✓ Cancel anytime
            </p>
          </motion.div>
        </div>
      </section>
    </>
  );
};

export default LandingPage;
