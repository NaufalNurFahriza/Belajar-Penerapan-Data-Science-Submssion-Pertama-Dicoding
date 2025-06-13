"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import {
  Bar,
  BarChart,
  XAxis,
  YAxis,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
  LineChart,
  Line,
} from "recharts"
import { Button } from "@/components/ui/button"
import { Download } from "lucide-react"

export default function HRDashboard() {
  const [activeTab, setActiveTab] = useState("overview")

  // Department attrition data
  const departmentData = [
    { name: "Sales", attrition: 20.8 },
    { name: "R&D", attrition: 13.2 },
    { name: "HR", attrition: 19.1 },
  ]

  // Job role attrition data
  const jobRoleData = [
    { name: "Sales Rep", attrition: 39.8 },
    { name: "HR Rep", attrition: 22.5 },
    { name: "Lab Tech", attrition: 19.7 },
    { name: "Manager", attrition: 15.3 },
    { name: "Developer", attrition: 10.2 },
  ]

  // Age group attrition data
  const ageGroupData = [
    { name: "18-25", attrition: 31.2 },
    { name: "26-35", attrition: 17.5 },
    { name: "36-45", attrition: 12.8 },
    { name: "46-55", attrition: 8.4 },
    { name: "56-65", attrition: 6.2 },
  ]

  // Overtime attrition data
  const overtimeData = [
    { name: "Yes", attrition: 30.5 },
    { name: "No", attrition: 10.2 },
  ]

  // Job satisfaction attrition data
  const satisfactionData = [
    { name: "Low (1)", attrition: 28.6 },
    { name: "Medium (2)", attrition: 17.3 },
    { name: "High (3)", attrition: 11.5 },
    { name: "Very High (4)", attrition: 8.2 },
  ]

  // Feature importance data
  const featureImportanceData = [
    { name: "Overtime", importance: 0.18 },
    { name: "Monthly Income", importance: 0.15 },
    { name: "Age", importance: 0.12 },
    { name: "Job Satisfaction", importance: 0.09 },
    { name: "Years at Company", importance: 0.08 },
  ]

  // Monthly attrition trend data
  const monthlyTrendData = [
    { month: "Jan", attrition: 12.5 },
    { month: "Feb", attrition: 13.2 },
    { month: "Mar", attrition: 14.1 },
    { month: "Apr", attrition: 15.3 },
    { month: "May", attrition: 16.2 },
    { month: "Jun", attrition: 15.8 },
    { month: "Jul", attrition: 14.9 },
    { month: "Aug", attrition: 14.2 },
    { month: "Sep", attrition: 13.7 },
    { month: "Oct", attrition: 13.1 },
    { month: "Nov", attrition: 12.8 },
    { month: "Dec", attrition: 12.3 },
  ]

  // Pie chart colors
  const COLORS = ["#0088FE", "#00C49F", "#FFBB28", "#FF8042", "#8884d8"]

  return (
    <div className="container mx-auto py-10">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">HR Analytics Dashboard</h1>
          <p className="text-muted-foreground">Jaya Jaya Maju - Employee Attrition Analysis</p>
        </div>
        <Button variant="outline" size="sm">
          <Download className="mr-2 h-4 w-4" />
          Export Report
        </Button>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4 mb-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Overall Attrition Rate</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">16.1%</div>
            <p className="text-xs text-muted-foreground">+2.1% from previous year</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Highest Department Attrition</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">Sales (20.8%)</div>
            <p className="text-xs text-muted-foreground">7.6% higher than company average</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Overtime Impact</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">+20.3%</div>
            <p className="text-xs text-muted-foreground">Increased attrition for overtime workers</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Top Attrition Factor</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">Overtime</div>
            <p className="text-xs text-muted-foreground">18% contribution to attrition</p>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="overview" className="space-y-4" onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="departments">Departments</TabsTrigger>
          <TabsTrigger value="demographics">Demographics</TabsTrigger>
          <TabsTrigger value="satisfaction">Satisfaction</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            <Card className="col-span-2">
              <CardHeader>
                <CardTitle>Monthly Attrition Trend</CardTitle>
                <CardDescription>Attrition rate over the past 12 months</CardDescription>
              </CardHeader>
              <CardContent className="pl-2">
                <ResponsiveContainer width="100%" height={350}>
                  <LineChart data={monthlyTrendData}>
                    <XAxis dataKey="month" />
                    <YAxis />
                    <ChartTooltip content={<ChartTooltipContent />} />
                    <Line
                      type="monotone"
                      dataKey="attrition"
                      stroke="#8884d8"
                      strokeWidth={2}
                      name="Attrition Rate (%)"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Top Attrition Factors</CardTitle>
                <CardDescription>Feature importance from predictive model</CardDescription>
              </CardHeader>
              <CardContent>
                <ChartContainer
                  config={{
                    importance: {
                      label: "Importance",
                      color: "hsl(var(--chart-1))",
                    },
                  }}
                  className="h-[350px]"
                >
                  <BarChart
                    data={featureImportanceData}
                    layout="vertical"
                    margin={{
                      left: 80,
                    }}
                  >
                    <XAxis type="number" />
                    <YAxis dataKey="name" type="category" tickLine={false} axisLine={false} />
                    <ChartTooltip content={<ChartTooltipContent />} />
                    <Bar dataKey="importance" fill="var(--color-importance)" radius={4} name="Importance Score" />
                  </BarChart>
                </ChartContainer>
              </CardContent>
            </Card>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Attrition by Job Role</CardTitle>
                <CardDescription>Top 5 job roles with highest attrition rates</CardDescription>
              </CardHeader>
              <CardContent>
                <ChartContainer
                  config={{
                    attrition: {
                      label: "Attrition Rate (%)",
                      color: "hsl(var(--chart-2))",
                    },
                  }}
                  className="h-[300px]"
                >
                  <BarChart
                    data={jobRoleData}
                    margin={{
                      bottom: 40,
                    }}
                  >
                    <XAxis dataKey="name" tickLine={false} axisLine={false} angle={-45} textAnchor="end" height={80} />
                    <YAxis />
                    <ChartTooltip content={<ChartTooltipContent />} />
                    <Bar dataKey="attrition" fill="var(--color-attrition)" radius={4} name="Attrition Rate (%)" />
                  </BarChart>
                </ChartContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Attrition by Overtime</CardTitle>
                <CardDescription>Impact of overtime on employee attrition</CardDescription>
              </CardHeader>
              <CardContent className="flex justify-center">
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={overtimeData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      outerRadius={100}
                      fill="#8884d8"
                      dataKey="attrition"
                      nameKey="name"
                      label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    >
                      {overtimeData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Legend />
                    <ChartTooltip />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="departments" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Attrition by Department</CardTitle>
                <CardDescription>Comparison of attrition rates across departments</CardDescription>
              </CardHeader>
              <CardContent>
                <ChartContainer
                  config={{
                    attrition: {
                      label: "Attrition Rate (%)",
                      color: "hsl(var(--chart-3))",
                    },
                  }}
                  className="h-[350px]"
                >
                  <BarChart data={departmentData}>
                    <XAxis dataKey="name" />
                    <YAxis />
                    <ChartTooltip content={<ChartTooltipContent />} />
                    <Bar dataKey="attrition" fill="var(--color-attrition)" radius={4} name="Attrition Rate (%)" />
                  </BarChart>
                </ChartContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Department Insights</CardTitle>
                <CardDescription>Key findings from department analysis</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <h4 className="font-medium">Sales Department</h4>
                    <p className="text-sm text-muted-foreground">
                      Highest attrition rate at 20.8%. Key factors include high overtime (45% of staff), lower job
                      satisfaction scores (avg 2.1/4), and lower average monthly income compared to R&D.
                    </p>
                  </div>
                  <div>
                    <h4 className="font-medium">R&D Department</h4>
                    <p className="text-sm text-muted-foreground">
                      Lowest attrition at 13.2%. Benefits from higher job satisfaction (avg 3.2/4), better work-life
                      balance scores, and higher average monthly income.
                    </p>
                  </div>
                  <div>
                    <h4 className="font-medium">HR Department</h4>
                    <p className="text-sm text-muted-foreground">
                      19.1% attrition rate. Contributing factors include limited career growth opportunities (avg 2.3
                      years since last promotion) and moderate job satisfaction scores (avg 2.7/4).
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="demographics" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Attrition by Age Group</CardTitle>
                <CardDescription>How age correlates with employee attrition</CardDescription>
              </CardHeader>
              <CardContent>
                <ChartContainer
                  config={{
                    attrition: {
                      label: "Attrition Rate (%)",
                      color: "hsl(var(--chart-4))",
                    },
                  }}
                  className="h-[350px]"
                >
                  <BarChart data={ageGroupData}>
                    <XAxis dataKey="name" />
                    <YAxis />
                    <ChartTooltip content={<ChartTooltipContent />} />
                    <Bar dataKey="attrition" fill="var(--color-attrition)" radius={4} name="Attrition Rate (%)" />
                  </BarChart>
                </ChartContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Demographic Insights</CardTitle>
                <CardDescription>Key findings from demographic analysis</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <h4 className="font-medium">Age Impact</h4>
                    <p className="text-sm text-muted-foreground">
                      Younger employees (18-25) show significantly higher attrition (31.2%) compared to older age
                      groups. Attrition decreases steadily with age, with the 56-65 group having the lowest rate at
                      6.2%.
                    </p>
                  </div>
                  <div>
                    <h4 className="font-medium">Gender Analysis</h4>
                    <p className="text-sm text-muted-foreground">
                      Minimal difference in attrition rates between genders (Male: 16.3%, Female: 15.9%), suggesting
                      gender is not a significant factor in employee retention.
                    </p>
                  </div>
                  <div>
                    <h4 className="font-medium">Education & Experience</h4>
                    <p className="text-sm text-muted-foreground">
                      Employees with fewer years of experience (0-2 years) show 24.7% attrition compared to 7.3% for
                      those with 10+ years. Higher education levels correlate with lower attrition rates.
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="satisfaction" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Attrition by Job Satisfaction</CardTitle>
                <CardDescription>Impact of job satisfaction on employee retention</CardDescription>
              </CardHeader>
              <CardContent>
                <ChartContainer
                  config={{
                    attrition: {
                      label: "Attrition Rate (%)",
                      color: "hsl(var(--chart-5))",
                    },
                  }}
                  className="h-[350px]"
                >
                  <BarChart data={satisfactionData}>
                    <XAxis dataKey="name" />
                    <YAxis />
                    <ChartTooltip content={<ChartTooltipContent />} />
                    <Bar dataKey="attrition" fill="var(--color-attrition)" radius={4} name="Attrition Rate (%)" />
                  </BarChart>
                </ChartContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Satisfaction Metrics</CardTitle>
                <CardDescription>Analysis of various satisfaction indicators</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <h4 className="font-medium">Job Satisfaction</h4>
                    <p className="text-sm text-muted-foreground">
                      Strong inverse correlation with attrition. Employees reporting low satisfaction (level 1) have
                      28.6% attrition rate, compared to just 8.2% for those with very high satisfaction (level 4).
                    </p>
                  </div>
                  <div>
                    <h4 className="font-medium">Work-Life Balance</h4>
                    <p className="text-sm text-muted-foreground">
                      Employees reporting poor work-life balance (level 1) show 27.3% attrition, while those with
                      excellent balance (level 4) have only 9.1% attrition.
                    </p>
                  </div>
                  <div>
                    <h4 className="font-medium">Environment Satisfaction</h4>
                    <p className="text-sm text-muted-foreground">
                      Low environment satisfaction correlates with 25.8% attrition rate, compared to 10.3% for high
                      satisfaction, highlighting the importance of workplace environment.
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
