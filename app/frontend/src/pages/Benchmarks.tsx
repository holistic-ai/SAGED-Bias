import React, { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogFooter,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Trash2, Edit, Plus } from "lucide-react";
import { useToast } from "@/components/ui/use-toast";

interface Benchmark {
  id: number;
  name: string;
  description: string;
  domain: string;
  concepts: string[];
  data_tier: string;
  status: string;
  is_active: boolean;
  created_at: string;
  created_by: string;
}

interface CreateBenchmarkData {
  name: string;
  description: string;
  domain: string;
  concepts: string[];
  data_tier: string;
  config: {
    concepts: string[];
    branching: boolean;
    branching_config?: {
      branching_pairs: 'not_all' | 'all';
      direction: 'both' | 'forward' | 'backward';
      source_restriction?: string;
      replacement_descriptor_require: boolean;
      descriptor_threshold: 'Auto' | number;
      descriptor_embedding_model: string;
      descriptor_distance: 'cosine' | 'euclidean';
      replacement_description: Record<string, any>;
      replacement_description_saving: boolean;
      replacement_description_saving_location?: string;
      counterfactual_baseline: boolean;
      generation_function?: any;
    };
    shared_config: {
      keyword_finder: {
        require: boolean;
        method: 'embedding_on_wiki' | 'llm_inquiries' | 'hyperlinks_on_wiki';
        keyword_number?: number;
        max_adjustment?: number;
        embedding_model?: string;
        saving: boolean;
        manual_keywords?: string[];
      };
      source_finder: {
        require: boolean;
        method: 'wiki' | 'local_files';
        local_file?: string;
        scrap_number?: number;
        saving: boolean;
        scrape_backlinks?: number;
      };
      scraper: {
        require: boolean;
        method: 'wiki' | 'local_files';
        saving: boolean;
      };
      prompt_assembler: {
        require: boolean;
        method: 'split_sentences' | 'questions';
        generation_function?: any;
        keyword_list?: string[];
        answer_check: boolean;
        max_benchmark_length: number;
      };
    };
    concept_specified_config?: Record<string, any>;
    saving: boolean;
  };
  created_by?: string;
}

const DOMAINS = [
  "employment",
  "healthcare",
  "education",
  "housing",
  "criminal-justice",
  "demographics",
  "professional",
];

const CONCEPTS = [
  "nationality",
  "gender",
  "race",
  "religion",
  "profession",
  "age",
];

const DATA_TIERS = [
  "lite",
  "keywords",
  "source_finder",
  "scraped_sentences",
  "split_sentences",
  "questions",
];

const fetchBenchmarks = async (): Promise<{ benchmarks: Benchmark[] }> => {
  const response = await fetch("/api/v1/benchmarks/");
  if (!response.ok) {
    throw new Error("Failed to fetch benchmarks");
  }
  return response.json();
};

const createBenchmark = async (
  data: CreateBenchmarkData
): Promise<Benchmark> => {
  const response = await fetch("/api/v1/benchmarks/", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  });
  if (!response.ok) {
    throw new Error("Failed to create benchmark");
  }
  return response.json();
};

const deleteBenchmark = async (id: number): Promise<void> => {
  const response = await fetch(`/api/v1/benchmarks/${id}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    throw new Error("Failed to delete benchmark");
  }
};

const Benchmarks: React.FC = () => {
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [formData, setFormData] = useState<CreateBenchmarkData>({
    name: "",
    description: "",
    domain: "",
    concepts: [],
    data_tier: "",
    config: {
      concepts: [],
      branching: false,
      shared_config: {
        keyword_finder: {
          require: true,
          method: 'embedding_on_wiki',
          keyword_number: 7,
          max_adjustment: 150,
          embedding_model: 'paraphrase-Mpnet-base-v2',
          saving: true
        },
        source_finder: {
          require: true,
          method: 'wiki',
          scrap_number: 5,
          saving: true,
          scrape_backlinks: 0
        },
        scraper: {
          require: true,
          method: 'wiki',
          saving: true
        },
        prompt_assembler: {
          require: true,
          method: 'split_sentences',
          answer_check: false,
          max_benchmark_length: 500
        }
      },
      saving: true
    },
    created_by: "user",
  });

  const queryClient = useQueryClient();
  const { toast } = useToast();

  const {
    data: benchmarksData,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["benchmarks"],
    queryFn: fetchBenchmarks,
  });

  const createMutation = useMutation({
    mutationFn: createBenchmark,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["benchmarks"] });
      setIsCreateModalOpen(false);
      setFormData({
        name: "",
        description: "",
        domain: "",
        concepts: [],
        data_tier: "",
        config: {
          concepts: [],
          branching: false,
          shared_config: {
            keyword_finder: {
              require: true,
              method: 'embedding_on_wiki',
              keyword_number: 7,
              max_adjustment: 150,
              embedding_model: 'paraphrase-Mpnet-base-v2',
              saving: true
            },
            source_finder: {
              require: true,
              method: 'wiki',
              scrap_number: 5,
              saving: true,
              scrape_backlinks: 0
            },
            scraper: {
              require: true,
              method: 'wiki',
              saving: true
            },
            prompt_assembler: {
              require: true,
              method: 'split_sentences',
              answer_check: false,
              max_benchmark_length: 500
            }
          },
          saving: true
        },
        created_by: "user",
      });
      toast({
        title: "Success",
        description: "Benchmark created successfully",
      });
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to create benchmark",
        variant: "destructive",
      });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: deleteBenchmark,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["benchmarks"] });
      toast({
        title: "Success",
        description: "Benchmark deleted successfully",
      });
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to delete benchmark",
        variant: "destructive",
      });
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (
      !formData.name ||
      !formData.domain ||
      formData.concepts.length === 0
    ) {
      toast({
        title: "Error",
        description: "Please fill in all required fields",
        variant: "destructive",
      });
      return;
    }

    // Update config.concepts to match the selected concepts
    const updatedFormData = {
      ...formData,
      config: {
        ...formData.config,
        concepts: formData.concepts
      }
    };

    createMutation.mutate(updatedFormData);
  };

  const handleConceptChange = (concept: string, checked: boolean) => {
    setFormData((prev) => ({
      ...prev,
      concepts: checked
        ? [...prev.concepts, concept]
        : prev.concepts.filter((c) => c !== concept),
    }));
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString();
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "ready":
        return "bg-green-100 text-green-800";
      case "processing":
        return "bg-yellow-100 text-yellow-800";
      case "draft":
        return "bg-gray-100 text-gray-800";
      case "complete":
        return "bg-blue-100 text-blue-800";
      default:
        return "bg-gray-100 text-gray-800";
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-96">
        <div className="text-lg">Loading benchmarks...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-96">
        <div className="text-lg text-red-600">Error loading benchmarks</div>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-4">
      <div className="mb-8">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Benchmarks</h1>
            <p className="mt-2 text-gray-600">
              Create and manage bias benchmarks for your experiments
            </p>
          </div>

          <Dialog open={isCreateModalOpen} onOpenChange={setIsCreateModalOpen}>
            <DialogTrigger asChild>
              <Button className="flex items-center gap-2">
                <Plus className="h-4 w-4" />
                Create Benchmark
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
              <DialogHeader>
                <DialogTitle>Create New Benchmark</DialogTitle>
              </DialogHeader>
              <form onSubmit={handleSubmit} className="space-y-4">
                {/* Basic Information */}
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold">Basic Information</h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="name">Name</Label>
                      <Input
                        id="name"
                        value={formData.name}
                        onChange={(e) =>
                          setFormData({ ...formData, name: e.target.value })
                        }
                        required
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="domain">Domain</Label>
                      <Input
                        id="domain"
                        value={formData.domain}
                        onChange={(e) =>
                          setFormData({ ...formData, domain: e.target.value })
                        }
                        required
                      />
                    </div>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="description">Description</Label>
                    <Textarea
                      id="description"
                      value={formData.description}
                      onChange={(e) =>
                        setFormData({ ...formData, description: e.target.value })
                      }
                      required
                    />
                  </div>
                </div>

                {/* Concepts Selection */}
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold">Bias Concepts</h3>
                  <div className="space-y-2">
                    <Label htmlFor="concepts">Concepts (comma-separated)</Label>
                    <Input
                      id="concepts"
                      value={formData.concepts.join(", ")}
                      onChange={(e) =>
                        setFormData({
                          ...formData,
                          concepts: e.target.value.split(",").map(c => c.trim()).filter(c => c !== "")
                        })
                      }
                      placeholder="Enter concepts separated by commas"
                      required
                    />
                  </div>
                </div>

                {/* Data Tier */}
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold">Data Tier</h3>
                  <div className="space-y-2">
                    <Label htmlFor="data_tier">Data Tier</Label>
                    <Input
                      id="data_tier"
                      value={formData.data_tier}
                      onChange={(e) =>
                        setFormData({ ...formData, data_tier: e.target.value })
                      }
                      required
                    />
                  </div>
                </div>

                {/* Configuration Settings */}
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold">Configuration Settings</h3>
                  
                  {/* Branching Configuration */}
                  <div className="space-y-2">
                    <div className="flex items-center space-x-2">
                      <Checkbox
                        id="branching"
                        checked={formData.config.branching}
                        onCheckedChange={(checked) =>
                          setFormData({
                            ...formData,
                            config: {
                              ...formData.config,
                              branching: checked as boolean,
                            },
                          })
                        }
                      />
                      <Label htmlFor="branching">Enable Branching</Label>
                    </div>
                  </div>

                  {/* Shared Configuration */}
                  <div className="space-y-4">
                    <h4 className="font-medium">Shared Configuration</h4>
                    
                    {/* Keyword Finder */}
                    <div className="space-y-2 border p-4 rounded-lg">
                      <h5 className="font-medium">Keyword Finder</h5>
                      <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <Label>Method</Label>
                          <Select
                            value={formData.config.shared_config.keyword_finder.method}
                            onValueChange={(value) =>
                              setFormData({
                                ...formData,
                                config: {
                                  ...formData.config,
                                  shared_config: {
                                    ...formData.config.shared_config,
                                    keyword_finder: {
                                      ...formData.config.shared_config.keyword_finder,
                                      method: value as any,
                                    },
                                  },
                                },
                              })
                            }
                          >
                            <SelectTrigger>
                              <SelectValue placeholder="Select method" />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="embedding_on_wiki">Embedding on Wiki</SelectItem>
                              <SelectItem value="llm_inquiries">LLM Inquiries</SelectItem>
                              <SelectItem value="hyperlinks_on_wiki">Hyperlinks on Wiki</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                        <div className="space-y-2">
                          <Label>Keyword Number</Label>
                          <Input
                            type="number"
                            value={formData.config.shared_config.keyword_finder.keyword_number}
                            onChange={(e) =>
                              setFormData({
                                ...formData,
                                config: {
                                  ...formData.config,
                                  shared_config: {
                                    ...formData.config.shared_config,
                                    keyword_finder: {
                                      ...formData.config.shared_config.keyword_finder,
                                      keyword_number: parseInt(e.target.value),
                                    },
                                  },
                                },
                              })
                            }
                          />
                        </div>
                      </div>
                    </div>

                    {/* Source Finder */}
                    <div className="space-y-2 border p-4 rounded-lg">
                      <h5 className="font-medium">Source Finder</h5>
                      <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <Label>Method</Label>
                          <Select
                            value={formData.config.shared_config.source_finder.method}
                            onValueChange={(value) =>
                              setFormData({
                                ...formData,
                                config: {
                                  ...formData.config,
                                  shared_config: {
                                    ...formData.config.shared_config,
                                    source_finder: {
                                      ...formData.config.shared_config.source_finder,
                                      method: value as any,
                                    },
                                  },
                                },
                              })
                            }
                          >
                            <SelectTrigger>
                              <SelectValue placeholder="Select method" />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="wiki">Wiki</SelectItem>
                              <SelectItem value="local_files">Local Files</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                        <div className="space-y-2">
                          <Label>Scrap Number</Label>
                          <Input
                            type="number"
                            value={formData.config.shared_config.source_finder.scrap_number}
                            onChange={(e) =>
                              setFormData({
                                ...formData,
                                config: {
                                  ...formData.config,
                                  shared_config: {
                                    ...formData.config.shared_config,
                                    source_finder: {
                                      ...formData.config.shared_config.source_finder,
                                      scrap_number: parseInt(e.target.value),
                                    },
                                  },
                                },
                              })
                            }
                          />
                        </div>
                      </div>
                    </div>

                    {/* Prompt Assembler */}
                    <div className="space-y-2 border p-4 rounded-lg">
                      <h5 className="font-medium">Prompt Assembler</h5>
                      <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <Label>Method</Label>
                          <Select
                            value={formData.config.shared_config.prompt_assembler.method}
                            onValueChange={(value) =>
                              setFormData({
                                ...formData,
                                config: {
                                  ...formData.config,
                                  shared_config: {
                                    ...formData.config.shared_config,
                                    prompt_assembler: {
                                      ...formData.config.shared_config.prompt_assembler,
                                      method: value as any,
                                    },
                                  },
                                },
                              })
                            }
                          >
                            <SelectTrigger>
                              <SelectValue placeholder="Select method" />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="split_sentences">Split Sentences</SelectItem>
                              <SelectItem value="questions">Questions</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                        <div className="space-y-2">
                          <Label>Max Benchmark Length</Label>
                          <Input
                            type="number"
                            value={formData.config.shared_config.prompt_assembler.max_benchmark_length}
                            onChange={(e) =>
                              setFormData({
                                ...formData,
                                config: {
                                  ...formData.config,
                                  shared_config: {
                                    ...formData.config.shared_config,
                                    prompt_assembler: {
                                      ...formData.config.shared_config.prompt_assembler,
                                      max_benchmark_length: parseInt(e.target.value),
                                    },
                                  },
                                },
                              })
                            }
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                <DialogFooter>
                  <Button type="submit" disabled={createMutation.isPending}>
                    {createMutation.isPending ? "Creating..." : "Create Benchmark"}
                  </Button>
                </DialogFooter>
              </form>
            </DialogContent>
          </Dialog>
        </div>
      </div>

      {/* Benchmarks List */}
      <div className="space-y-4">
        {benchmarksData?.benchmarks?.length === 0 ? (
          <Card>
            <CardContent className="flex flex-col items-center justify-center py-12">
              <p className="text-gray-500 text-lg mb-4">No benchmarks found</p>
              <p className="text-gray-400 text-sm">
                Create your first benchmark to get started
              </p>
            </CardContent>
          </Card>
        ) : (
          benchmarksData?.benchmarks?.map((benchmark) => (
            <Card key={benchmark.id}>
              <CardContent className="p-6">
                <div className="flex justify-between items-start">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <h3 className="text-lg font-semibold text-gray-900">
                        {benchmark.name}
                      </h3>
                      <Badge className={getStatusColor(benchmark.status)}>
                        {benchmark.status}
                      </Badge>
                    </div>

                    {benchmark.description && (
                      <p className="text-gray-600 mb-2">
                        {benchmark.description}
                      </p>
                    )}

                    <div className="flex items-center gap-4 text-sm text-gray-500">
                      <span>Domain: {benchmark.domain}</span>
                      <span>•</span>
                      <span>Concepts: {benchmark.concepts.join(", ")}</span>
                      <span>•</span>
                      <span>Created {formatDate(benchmark.created_at)}</span>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    <Button variant="outline" size="sm">
                      <Edit className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => deleteMutation.mutate(benchmark.id)}
                      disabled={deleteMutation.isPending}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))
        )}
      </div>
    </div>
  );
};

export default Benchmarks;
