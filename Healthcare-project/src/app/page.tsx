import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Heart, Shield, Users, Calendar, Activity, Stethoscope, ArrowRight, CheckCircle } from "lucide-react";

const features = [
  { icon: Stethoscope, title: "Doctor Dashboard", desc: "Manage patients, diagnoses, prescriptions, and appointments in one place." },
  { icon: Users, title: "Patient Portal", desc: "Access records, book appointments, and communicate with your care team." },
  { icon: Shield, title: "Admin Control", desc: "Full system oversight with analytics, user management, and billing." },
  { icon: Activity, title: "AI-Assisted Care", desc: "Ollama-powered diagnosis support and automated patient summaries." },
  { icon: Calendar, title: "Smart Scheduling", desc: "Intelligent appointment management with n8n workflow automations." },
  { icon: Heart, title: "Secure & Compliant", desc: "Built on Supabase with row-level security and end-to-end encryption." },
];

const highlights = [
  "50+ patient records manageable simultaneously",
  "Real-time notifications and reminders",
  "Document upload and secure storage",
  "AI-powered diagnosis support",
  "Automated workflow with n8n",
  "Responsive across all devices",
];

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-white">
      <header className="border-b border-gray-100 bg-white sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="h-8 w-8 rounded-lg bg-blue-600 flex items-center justify-center">
              <Heart className="h-4 w-4 text-white" />
            </div>
            <span className="font-bold text-gray-900 text-lg">MediCare Pro</span>
          </div>
          <nav className="hidden md:flex items-center gap-6 text-sm text-gray-600">
            <a href="#features" className="hover:text-gray-900 transition-colors">Features</a>
            <a href="#roles" className="hover:text-gray-900 transition-colors">Dashboards</a>
          </nav>
          <div className="flex items-center gap-3">
            <Link href="/login"><Button variant="ghost" size="sm">Sign In</Button></Link>
            <Link href="/register"><Button size="sm">Get Started</Button></Link>
          </div>
        </div>
      </header>

      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-20 pb-16 text-center">
        <div className="inline-flex items-center gap-2 bg-blue-50 text-blue-700 rounded-full px-4 py-1.5 text-sm font-medium mb-6">
          <Activity className="h-3.5 w-3.5" />
          Healthcare Management Platform
        </div>
        <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-gray-900 leading-tight mb-6">
          Modern Healthcare,<br />
          <span className="text-blue-600">Simplified.</span>
        </h1>
        <p className="text-lg text-gray-500 max-w-2xl mx-auto mb-10">
          A complete healthcare management system for doctors, patients, and administrators. AI-assisted care, smart scheduling, and secure records — all in one platform.
        </p>
        <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
          <Link href="/register">
            <Button size="lg" className="gap-2">Start Free <ArrowRight className="h-4 w-4" /></Button>
          </Link>
          <Link href="/login">
            <Button size="lg" variant="outline">Sign In to Dashboard</Button>
          </Link>
        </div>
      </section>

      <section id="roles" className="bg-gray-50 py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-2xl font-bold text-gray-900 text-center mb-10">Three Dashboards, One Platform</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[
              { role: "Doctor", href: "/doctor", bg: "bg-blue-600", light: "bg-blue-50", text: "text-blue-600", desc: "Manage patients, view records, write prescriptions, and track appointments.", icon: Stethoscope },
              { role: "Patient", href: "/patient", bg: "bg-green-600", light: "bg-green-50", text: "text-green-600", desc: "Book appointments, view medical history, download documents, and message your doctor.", icon: Heart },
              { role: "Admin", href: "/admin", bg: "bg-purple-600", light: "bg-purple-50", text: "text-purple-600", desc: "Manage users, view analytics, control billing, and configure system settings.", icon: Shield },
            ].map(({ role, href, bg, light, text, desc, icon: Icon }) => (
              <div key={role} className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6 hover:shadow-md transition-shadow">
                <div className={`inline-flex p-3 rounded-xl ${light} mb-4`}>
                  <Icon className={`h-6 w-6 ${text}`} />
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">{role} Dashboard</h3>
                <p className="text-sm text-gray-500 mb-5">{desc}</p>
                <Link href={href}>
                  <Button className={`w-full ${bg} hover:opacity-90 text-white`} size="sm">
                    Open {role} Dashboard
                  </Button>
                </Link>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section id="features" className="py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-2xl font-bold text-gray-900 text-center mb-3">Everything You Need</h2>
          <p className="text-gray-500 text-center mb-10">Production-ready features for real-world healthcare usage</p>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map(({ icon: Icon, title, desc }) => (
              <div key={title} className="p-5 rounded-xl border border-gray-100 bg-white hover:border-blue-100 hover:shadow-sm transition-all">
                <div className="p-2.5 bg-blue-50 rounded-lg w-fit mb-4">
                  <Icon className="h-5 w-5 text-blue-600" />
                </div>
                <h3 className="font-semibold text-gray-900 mb-1">{title}</h3>
                <p className="text-sm text-gray-500">{desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="bg-blue-600 py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 items-center">
            <div>
              <h2 className="text-2xl font-bold text-white mb-4">Built for Real Healthcare Workflows</h2>
              <p className="text-blue-100 mb-6">Every feature is designed to match actual clinical and administrative workflows.</p>
              <Link href="/register">
                <Button variant="outline" className="bg-white text-blue-600 hover:bg-blue-50 border-white">
                  Get Started Today
                </Button>
              </Link>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {highlights.map((item) => (
                <div key={item} className="flex items-start gap-2.5 text-white">
                  <CheckCircle className="h-4 w-4 text-blue-200 mt-0.5 shrink-0" />
                  <span className="text-sm">{item}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      <footer className="border-t border-gray-100 py-8 text-center text-sm text-gray-400">
        <div className="flex items-center justify-center gap-2 mb-2">
          <div className="h-6 w-6 rounded-md bg-blue-600 flex items-center justify-center">
            <Heart className="h-3 w-3 text-white" />
          </div>
          <span className="font-semibold text-gray-900">MediCare Pro</span>
        </div>
        <p>© {new Date().getFullYear()} MediCare Pro. Built with Next.js, Supabase, Tailwind CSS, and Ollama.</p>
      </footer>
    </div>
  );
}
