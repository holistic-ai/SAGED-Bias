import React, { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
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
  categories: string[];
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
  categories: string[];
  data_tier: string;
  config: Record<string, any>;
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

const CATEGORIES = [
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
    categories: [],
    data_tier: "lite",
    config: { model: "default", temperature: 0.7 },
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
        categories: [],
        data_tier: "lite",
        config: { model: "default", temperature: 0.7 },
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
      formData.categories.length === 0
    ) {
      toast({
        title: "Error",
        description: "Please fill in all required fields",
        variant: "destructive",
      });
      return;
    }
    createMutation.mutate(formData);
  };

  const handleCategoryChange = (category: string, checked: boolean) => {
    setFormData((prev) => ({
      ...prev,
      categories: checked
        ? [...prev.categories, category]
        : prev.categories.filter((c) => c !== category),
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
    <div>
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
            <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
              <DialogHeader>
                <DialogTitle>Create New Benchmark</DialogTitle>
              </DialogHeader>
              <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                  <Label htmlFor="name">Name *</Label>
                  <Input
                    id="name"
                    value={formData.name}
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                      setFormData((prev) => ({ ...prev, name: e.target.value }))
                    }
                    placeholder="Enter benchmark name"
                    required
                  />
                </div>

                <div>
                  <Label htmlFor="description">Description</Label>
                  <Textarea
                    id="description"
                    value={formData.description}
                    onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) =>
                      setFormData((prev) => ({
                        ...prev,
                        description: e.target.value,
                      }))
                    }
                    placeholder="Describe your benchmark"
                  />
                </div>

                <div>
                  <Label htmlFor="domain">Domain *</Label>
                  <Select
                    value={formData.domain}
                    onValueChange={(value: string) =>
                      setFormData((prev) => ({ ...prev, domain: value }))
                    }
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select a domain" />
                    </SelectTrigger>
                    <SelectContent>
                      {DOMAINS.map((domain) => (
                        <SelectItem key={domain} value={domain}>
                          {domain.charAt(0).toUpperCase() + domain.slice(1)}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <Label>Bias Categories *</Label>
                  <div className="grid grid-cols-2 gap-2 mt-2">
                    {CATEGORIES.map((category) => (
                      <div
                        key={category}
                        className="flex items-center space-x-2"
                      >
                        <Checkbox
                          id={category}
                          checked={formData.categories.includes(category)}
                          onCheckedChange={(checked: boolean) =>
                            handleCategoryChange(category, checked)
                          }
                        />
                        <Label htmlFor={category} className="text-sm">
                          {category.charAt(0).toUpperCase() + category.slice(1)}
                        </Label>
                      </div>
                    ))}
                  </div>
                </div>

                <div>
                  <Label htmlFor="data_tier">Data Tier</Label>
                  <Select
                    value={formData.data_tier}
                    onValueChange={(value: string) =>
                      setFormData((prev) => ({ ...prev, data_tier: value }))
                    }
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {DATA_TIERS.map((tier) => (
                        <SelectItem key={tier} value={tier}>
                          {tier.charAt(0).toUpperCase() + tier.slice(1)}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="flex justify-end space-x-2 pt-4">
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => setIsCreateModalOpen(false)}
                  >
                    Cancel
                  </Button>
                  <Button type="submit" disabled={createMutation.isPending}>
                    {createMutation.isPending
                      ? "Creating..."
                      : "Create Benchmark"}
                  </Button>
                </div>
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
                      <span>Categories: {benchmark.categories.join(", ")}</span>
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
