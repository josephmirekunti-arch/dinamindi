import { useState, useEffect } from "react";
import { Info, TrendingUp, TrendingDown, Scale, Target, Activity, Loader2 } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Accordion, AccordionItem, AccordionTrigger, AccordionContent } from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import React from 'react';

interface ErrorBoundaryProps {
  children: React.ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
}

class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(_: any): ErrorBoundaryState {
    return { hasError: true };
  }

  componentDidCatch(error: any, errorInfo: any) {
    console.error("ErrorBoundary caught an error:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <Card className="border-red-200 bg-red-50/50 shadow-sm flex items-center justify-center p-6">
          <div className="text-center">
            <Info className="w-8 h-8 text-red-500 mx-auto mb-2" />
            <h3 className="text-sm font-bold text-red-900">Failed to Parse Match</h3>
            <p className="text-xs text-red-700/80 mt-1">This fixture was omitted due to corrupted data.</p>
          </div>
        </Card>
      );
    }
    return this.props.children;
  }
}

export default function App() {
  const [, setActiveTab] = useState("predictions");
  const [predictData, setPredictData] = useState([]);
  const [perfData, setPerfData] = useState([]);
  const [perfSummary, setPerfSummary] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchData(false);
    const interval = setInterval(() => {
      fetchData(true); // Silent poll every 60 seconds
    }, 60000);
    return () => clearInterval(interval);
  }, []);

  const fetchData = async (silent = false) => {
    try {
      if (!silent) setLoading(true);

      // Use VITE_API_BASE_URL from Vercel environment, fallback to localhost for local dev
      const baseUrl = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8004";

      const [predRes, perfRes] = await Promise.all([
        fetch(`${baseUrl}/api/predictions`),
        fetch(`${baseUrl}/api/performance`)
      ]);
      const predJson = await predRes.json();
      const perfJson = await perfRes.json();
      setPredictData(predJson.data || []);
      setPerfData(perfJson.data || []);
      setPerfSummary(perfJson.summary || []);
    } catch (e) {
      console.error(e);
    } finally {
      if (!silent) setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-zinc-50 dark:bg-zinc-950">
        <Loader2 className="w-10 h-10 text-primary animate-spin" />
        <p className="mt-4 text-zinc-600 font-medium animate-pulse">Running Poisson Engine Models...</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-zinc-50 text-zinc-900 p-4 md:p-8">
      <div className="max-w-6xl mx-auto space-y-8">

        {/* Header */}
        <header className="flex flex-col md:flex-row justify-between items-start md:items-center pb-6 border-b border-zinc-200">
          <div>
            <h1 className="text-3xl font-extrabold tracking-tight flex items-center gap-2">
              <Activity className="w-8 h-8 text-primary" /> Dinamindi
            </h1>
            <p className="text-sm text-zinc-500 mt-1 font-medium">Institutional Grade Football Quant Models</p>
          </div>
        </header>

        {/* Shadcn Tabs Navigation */}
        <Tabs defaultValue="predictions" className="w-full" onValueChange={setActiveTab}>
          <TabsList className="w-full justify-start border-b rounded-none bg-transparent p-0 h-12 gap-8">
            <TabsTrigger
              value="predictions"
              className="data-[state=active]:border-b-2 data-[state=active]:border-primary data-[state=active]:shadow-none rounded-none px-0 bg-transparent h-full font-semibold text-base"
            >
              <Target className="w-4 h-4 mr-2" /> Live Markets (+EV)
            </TabsTrigger>
            <TabsTrigger
              value="performance"
              className="data-[state=active]:border-b-2 data-[state=active]:border-primary data-[state=active]:shadow-none rounded-none px-0 bg-transparent h-full font-semibold text-base"
            >
              <Scale className="w-4 h-4 mr-2" /> Model Performance
            </TabsTrigger>
          </TabsList>

          <div className="mt-8">
            {/* Predictions Tab Content */}
            <TabsContent value="predictions" className="mt-0 outline-none">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                {predictData.length === 0 ? (
                  <div className="col-span-2 text-center py-12 text-zinc-500 font-medium">No upcoming fixtures found.</div>
                ) : (
                  predictData.map((match: any, i: number) => (
                    <ErrorBoundary key={i}>
                      <Card className="border-zinc-200/60 shadow-sm hover:shadow-md transition-shadow">
                        <CardHeader className="px-6 pt-6 pb-4">
                          <div className="flex justify-between w-full items-center mb-6">
                            <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-zinc-400">
                              {match.competition === 'EPL' && <span className="text-sm">🇬🇧</span>}
                              {match.competition === 'La_Liga' && <span className="text-sm">🇪🇸</span>}
                              {match.competition === 'Bundesliga' && <span className="text-sm">🇩🇪</span>}
                              <span>
                                {new Date(match.date).toLocaleString(undefined, {
                                  weekday: 'short', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
                                })}
                              </span>
                            </div>
                            <Badge variant="secondary" className="font-mono text-[10px] tracking-widest">PRE-MATCH</Badge>
                          </div>

                          <div className="flex justify-between items-center w-full">
                            <div className="flex flex-col items-start w-[45%]">
                              <CardTitle className="text-xl leading-tight flex items-center gap-3">
                                {match.home_logo && <img src={match.home_logo} alt={match.home_team} className="w-8 h-8 object-contain" />}
                                {match.home_team}
                              </CardTitle>
                              <span className="text-sm text-zinc-500 font-mono mt-1 font-medium">xG: {match.lambda_h.toFixed(2)} &bull; ELO: {match.home_elo}</span>
                            </div>
                            <div className="text-2xl font-black text-zinc-200 text-center px-4">VS</div>
                            <div className="flex flex-col items-end w-[45%] text-right">
                              <CardTitle className="text-xl leading-tight flex items-center gap-3 justify-end w-full">
                                {match.away_team}
                                {match.away_logo && <img src={match.away_logo} alt={match.away_team} className="w-8 h-8 object-contain" />}
                              </CardTitle>
                              <span className="text-sm text-zinc-500 font-mono mt-1 font-medium">ELO: {match.away_elo} &bull; xG: {match.lambda_a.toFixed(2)}</span>
                            </div>
                          </div>
                        </CardHeader>

                        <Separator className="bg-zinc-100" />

                        <CardContent className="px-6 pb-6 pt-4">
                          {/* Elo Technical Insights */}
                          {match.elo_insights && (
                            <div className="mb-6 bg-indigo-50/50 rounded-xl p-4 border border-indigo-100">
                              <h4 className="text-[10px] font-bold text-indigo-800 uppercase tracking-widest mb-2 flex items-center gap-2">
                                <Info className="w-3 h-3" /> Technical Insight: {match.elo_insights.verdict}
                              </h4>
                              <p className="text-sm text-indigo-900/80 leading-relaxed mb-3">
                                {match.elo_insights.analysis}
                              </p>
                              <div className="flex flex-wrap gap-2">
                                {match.elo_insights.recommended_markets.map((m: string, idx: number) => (
                                  <Badge key={idx} variant="outline" className="text-[10px] bg-white border-indigo-200 text-indigo-700">
                                    {m}
                                  </Badge>
                                ))}
                              </div>
                            </div>
                          )}

                          <h4 className="text-xs font-bold text-zinc-400 uppercase tracking-widest mb-4">Calculated Edges</h4>
                          <Accordion type="multiple" className="w-full space-y-3">
                            {match.markets.map((m: any, j: number) => {
                              const isEv = m.ev_margin > 0;
                              const isSecondary = m.market.includes("CORNERS") || m.market.includes("CARDS");
                              const isRecommended = m.risk === 'recommended';

                              let borderClass = 'border-zinc-200 bg-white';
                              if (isSecondary) {
                                borderClass = 'border-indigo-100 bg-indigo-50/20';
                              } else if (isRecommended) {
                                borderClass = 'border-emerald-200 bg-emerald-50/10';
                              } else {
                                borderClass = 'border-rose-100 bg-rose-50/10';
                              }

                              return (
                                <AccordionItem
                                  key={j}
                                  value={`market-${i}-${j}`}
                                  className={`border rounded-xl px-4 overflow-hidden transition-all duration-300 ${borderClass}`}
                                >
                                  <AccordionTrigger className="py-4 hover:no-underline">
                                    <div className="flex items-center justify-between w-full pr-4">
                                      <div className="flex items-center gap-3">
                                        <span className={`text-sm font-black tracking-tight ${isSecondary
                                          ? 'text-indigo-700'
                                          : (isRecommended ? 'text-emerald-700' : 'text-rose-700')
                                          }`}>
                                          {m.market}
                                        </span>
                                        {isEv && !isSecondary && (
                                          <Badge variant="outline" className="text-[10px] font-bold bg-amber-100 text-amber-800 border-amber-300">
                                            MV +{(m.ev_margin * 100).toFixed(1)}%
                                          </Badge>
                                        )}
                                        {isSecondary && (
                                          <Badge variant="outline" className="text-[10px] font-bold bg-indigo-100 text-indigo-800 border-indigo-300">
                                            SECONDARY
                                          </Badge>
                                        )}
                                      </div>
                                      <span className="font-mono text-sm font-semibold text-zinc-700">
                                        {(m.probability * 100).toFixed(1)}%
                                      </span>
                                    </div>
                                  </AccordionTrigger>
                                  <AccordionContent className="pb-3 text-zinc-600">
                                    <div className="flex items-start gap-2 leading-relaxed text-sm">
                                      <Info className="w-4 h-4 mt-0.5 shrink-0 text-zinc-400" />
                                      {m.analysis}
                                    </div>

                                    {/* Value Betting Component for Match Winner/Goals only */}
                                    {m.bookie_odd && !isSecondary && (
                                      <div className="mt-3 bg-white rounded-xl p-4 border border-zinc-200 flex flex-col gap-3 shadow-sm">
                                        <div className="flex justify-between text-xs text-zinc-500 font-bold uppercase tracking-wider">
                                          <span>Bookmaker (Implied)</span>
                                          <span>Our Model</span>
                                        </div>
                                        <div className="flex justify-between items-center">
                                          <div className="flex items-center gap-2">
                                            <span className="font-mono font-bold text-lg text-zinc-800">{m.bookie_odd.toFixed(2)}</span>
                                            <span className="text-xs font-semibold text-zinc-400">({(m.implied_prob * 100).toFixed(1)}%)</span>
                                          </div>

                                          <div className="flex flex-col items-center">
                                            {isEv ? (
                                              <TrendingUp className="w-5 h-5 text-amber-500" />
                                            ) : (
                                              <TrendingDown className="w-5 h-5 text-zinc-300" />
                                            )}
                                          </div>

                                          <div className="font-mono font-bold text-lg text-zinc-800">
                                            {(m.probability * 100).toFixed(1)}%
                                          </div>
                                        </div>
                                        <div className={`mt-2 text-xs font-bold text-center py-1.5 rounded-md ${isEv ? 'bg-amber-50 text-amber-800' : 'bg-zinc-100 text-zinc-500'}`}>
                                          Verdict: {m.ev_status}
                                        </div>
                                      </div>
                                    )}
                                  </AccordionContent>
                                </AccordionItem>
                              );
                            })}
                          </Accordion>

                          {/* Goal Timing Analysis */}
                          {match.goal_intervals && (
                            <div className="mt-8 border-t border-zinc-100 pt-6">
                              <h4 className="text-[10px] font-bold text-zinc-400 uppercase tracking-widest mb-5 flex items-center gap-2">
                                <Activity className="w-3 h-3" /> Time of Goal Heatmap (10m Intervals)
                              </h4>

                              <div className="space-y-5">
                                {/* Home Team Timing */}
                                <div className="space-y-2">
                                  <div className="flex justify-between items-center mb-1">
                                    <span className="text-xs font-bold text-zinc-700">{match.home_team}</span>
                                    <span className="text-[10px] uppercase font-bold text-zinc-400">Scoring Prob</span>
                                  </div>
                                  <div className="flex h-8 w-full bg-zinc-50 rounded-md overflow-hidden border border-zinc-200 shadow-inner">
                                    {match.goal_intervals.home["10m"].map((prob: number, k: number) => (
                                      <div
                                        key={k}
                                        className="h-full relative group transition-all"
                                        style={{
                                          width: "10%",
                                          backgroundColor: `rgba(59, 130, 246, ${Math.max(0.05, prob * 3)})`,
                                          borderRight: k < 9 ? '1px solid rgba(0,0,0,0.05)' : 'none'
                                        }}
                                      >
                                        <div className="absolute inset-0 opacity-0 group-hover:opacity-100 bg-blue-500/20 flex items-center justify-center pointer-events-none transition-opacity">
                                          <span className="text-[9px] font-bold text-blue-900">{(prob * 100).toFixed(0)}%</span>
                                        </div>
                                      </div>
                                    ))}
                                  </div>
                                </div>

                                {/* Away Team Timing */}
                                <div className="space-y-2">
                                  <div className="flex justify-between items-center mb-1">
                                    <span className="text-xs font-bold text-zinc-700">{match.away_team}</span>
                                  </div>
                                  <div className="flex h-8 w-full bg-zinc-50 rounded-md overflow-hidden border border-zinc-200 shadow-inner">
                                    {match.goal_intervals.away["10m"].map((prob: number, k: number) => (
                                      <div
                                        key={k}
                                        className="h-full relative group transition-all"
                                        style={{
                                          width: "10%",
                                          backgroundColor: `rgba(168, 85, 247, ${Math.max(0.05, prob * 3)})`,
                                          borderRight: k < 9 ? '1px solid rgba(0,0,0,0.05)' : 'none'
                                        }}
                                      >
                                        <div className="absolute inset-0 opacity-0 group-hover:opacity-100 bg-purple-500/20 flex items-center justify-center pointer-events-none transition-opacity">
                                          <span className="text-[9px] font-bold text-purple-900">{(prob * 100).toFixed(0)}%</span>
                                        </div>
                                      </div>
                                    ))}
                                  </div>
                                </div>

                                <div className="flex justify-between text-[9px] text-zinc-400 font-bold px-1 tracking-wider uppercase mt-2">
                                  <span>0'</span>
                                  <span>15'</span>
                                  <span>30'</span>
                                  <span>45'</span>
                                  <span>60'</span>
                                  <span>75'</span>
                                  <span>90'+</span>
                                </div>
                              </div>
                            </div>
                          )}
                        </CardContent>
                      </Card>
                    </ErrorBoundary>
                  ))
                )}
              </div>
            </TabsContent>

            {/* Performance Tab Content */}
            <TabsContent value="performance" className="mt-0 outline-none">
              {/* Performance Summary */}
              {perfSummary.length > 0 && (
                <Card className="mb-8 border-indigo-100 bg-indigo-50/30 overflow-hidden shadow-sm">
                  <CardHeader className="py-4 border-b border-indigo-100/50 bg-indigo-50/50">
                    <CardTitle className="text-sm font-bold text-indigo-900 flex items-center gap-2">
                      <Target className="w-4 h-4 text-indigo-600" /> Model Accuracy (Past 10 Matches)
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="p-0">
                    <div className="grid grid-cols-2 md:grid-cols-4 divide-y md:divide-y-0 md:divide-x divide-indigo-100">
                      {perfSummary.slice(0, 4).map((mkt: any, idx: number) => (
                        <div key={idx} className="p-4 flex flex-col items-center justify-center text-center">
                          <span className="text-[10px] font-bold text-indigo-500 uppercase tracking-widest mb-1">{mkt.market}</span>
                          <span className="text-2xl font-black text-indigo-950 mb-1">{mkt.accuracy.toFixed(1)}%</span>
                          <span className="text-xs font-semibold text-indigo-400">{mkt.won}W / {mkt.total} Stakes</span>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )}

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {perfData.length === 0 ? (
                  <div className="col-span-2 text-center py-12 text-zinc-500 font-medium">No historical data available yet.</div>
                ) : (
                  perfData.map((match: any, i: number) => {
                    const wonCount = match.recommended.filter((r: any) => r.status === "WON").length;
                    const totalCount = match.recommended.length;
                    const acc = totalCount > 0 ? (wonCount / totalCount) * 100 : 0;

                    return (
                      <ErrorBoundary key={i}>
                        <Card className="border-zinc-200 shadow-sm overflow-hidden flex flex-col sm:flex-row">
                          {/* Left Side: Match Info */}
                          <div className="flex-1 p-6 border-b sm:border-b-0 sm:border-r border-zinc-100 bg-zinc-50/50 flex flex-col justify-center">
                            <div className="flex items-center gap-2 text-[10px] font-bold text-zinc-400 mb-4 uppercase tracking-widest">
                              {match.competition === 'EPL' && <span className="text-[#aeb1b5] text-xs">🇬🇧</span>}
                              {match.competition === 'La_Liga' && <span className="text-[#aeb1b5] text-xs">🇪🇸</span>}
                              {match.competition === 'Bundesliga' && <span className="text-[#aeb1b5] text-xs">🇩🇪</span>}
                              <span>{new Date(match.date).toLocaleDateString()}</span>
                            </div>
                            <div className="flex justify-between items-center mb-2">
                              <div className="flex items-center gap-2">
                                {match.home_logo && <img src={match.home_logo} alt={match.home_team} className="w-5 h-5 object-contain" />}
                                <span className="font-bold text-lg text-zinc-800">{match.home_team}</span>
                              </div>
                              <span className="font-mono font-black text-xl">{match.score.split('-')[0].trim()}</span>
                            </div>
                            <div className="flex justify-between items-center">
                              <div className="flex items-center gap-2">
                                {match.away_logo && <img src={match.away_logo} alt={match.away_team} className="w-5 h-5 object-contain" />}
                                <span className="font-bold text-lg text-zinc-500">{match.away_team}</span>
                              </div>
                              <span className="font-mono font-black text-xl text-zinc-500">{match.score.split('-')[1].trim()}</span>
                            </div>

                            <div className="mt-6 pt-6 border-t border-zinc-200/60 flex justify-between items-center">
                              <span className="text-xs text-zinc-500 font-bold uppercase tracking-wider">Model Strike Rate</span>
                              <Badge variant={acc >= 50 ? "default" : "destructive"} className={`font-mono text-sm py-0.5 ${acc >= 50 ? 'bg-green-600 hover:bg-green-700' : ''}`}>
                                {acc.toFixed(0)}%
                              </Badge>
                            </div>
                          </div>

                          {/* Right Side: Bets */}
                          <div className="flex-1 p-6 bg-white">
                            <h5 className="text-[10px] font-bold text-zinc-400 uppercase tracking-widest mb-4">Model Calls</h5>
                            <div className="flex flex-col gap-3">
                              {match.recommended.length === 0 ? (
                                <span className="text-sm text-zinc-400 italic font-medium">No stakes recommended.</span>
                              ) : (
                                match.recommended.map((rec: any, j: number) => (
                                  <div key={j} className="flex justify-between items-center py-1">
                                    <span className="text-sm font-bold text-zinc-700">{rec.market}</span>
                                    <div className="flex items-center gap-2">
                                      <div className={`w-2 h-2 rounded-full ${rec.status === 'WON' ? 'bg-green-500' : 'bg-red-500'}`} />
                                      <span className={`text-[10px] font-black tracking-widest ${rec.status === 'WON' ? 'text-green-700' : 'text-red-700'}`}>
                                        {rec.status}
                                      </span>
                                    </div>
                                  </div>
                                ))
                              )}
                            </div>
                          </div>
                        </Card>
                      </ErrorBoundary>
                    )
                  })
                )}
              </div>
            </TabsContent>
          </div>
        </Tabs>
      </div>
    </div>
  );
}
