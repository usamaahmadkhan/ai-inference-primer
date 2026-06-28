terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ─────────────────────────────────────────────
# EKS Cluster
# ─────────────────────────────────────────────

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = var.cluster_name
  cluster_version = var.kubernetes_version

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  # Allow public API endpoint for kubectl access
  # Restrict to your corp CIDR in production
  cluster_endpoint_public_access       = true
  cluster_endpoint_public_access_cidrs = var.allowed_cidr_blocks

  # Enable IRSA (IAM Roles for Service Accounts)
  enable_irsa = true

  # Managed node group: small on-demand CPU nodes for system workloads
  # GPU nodes are managed by Karpenter
  eks_managed_node_groups = {
    system = {
      name           = "system"
      instance_types = ["m5.xlarge"]
      min_size       = 2
      max_size       = 4
      desired_size   = 2

      labels = {
        workload-type = "system"
      }

      taints = []
    }
  }

  # Required add-ons
  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
      configuration_values = jsonencode({
        env = {
          ENABLE_PREFIX_DELEGATION = "true"   # More IPs per node
          WARM_PREFIX_TARGET       = "1"
        }
      })
    }
    aws-ebs-csi-driver = {
      most_recent              = true
      service_account_role_arn = module.ebs_csi_irsa_role.iam_role_arn
    }
  }

  tags = local.common_tags
}

# ─────────────────────────────────────────────
# Karpenter
# ─────────────────────────────────────────────

module "karpenter" {
  source  = "terraform-aws-modules/eks/aws//modules/karpenter"
  version = "~> 20.0"

  cluster_name = module.eks.cluster_name

  # Karpenter needs these permissions to provision GPU instances
  node_iam_role_additional_policies = {
    AmazonSSMManagedInstanceCore = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
  }

  tags = local.common_tags
}

resource "helm_release" "karpenter" {
  namespace        = "karpenter"
  create_namespace = true
  name             = "karpenter"
  repository       = "oci://public.ecr.aws/karpenter"
  chart            = "karpenter"
  version          = var.karpenter_version

  values = [
    <<-EOT
    settings:
      clusterName: ${module.eks.cluster_name}
      clusterEndpoint: ${module.eks.cluster_endpoint}
      interruptionQueue: ${module.karpenter.queue_name}
    serviceAccount:
      annotations:
        eks.amazonaws.com/role-arn: ${module.karpenter.iam_role_arn}
    EOT
  ]

  depends_on = [module.eks]
}

# ─────────────────────────────────────────────
# EFS for model cache (shared across pods)
# ─────────────────────────────────────────────

resource "aws_efs_file_system" "model_cache" {
  creation_token = "${var.cluster_name}-model-cache"
  encrypted      = true

  performance_mode = "generalPurpose"
  throughput_mode  = "elastic"       # Auto-scales throughput

  lifecycle_policy {
    transition_to_ia = "AFTER_30_DAYS"
  }

  tags = merge(local.common_tags, {
    Name = "${var.cluster_name}-model-cache"
  })
}

resource "aws_efs_mount_target" "model_cache" {
  for_each = toset(module.vpc.private_subnets)

  file_system_id  = aws_efs_file_system.model_cache.id
  subnet_id       = each.value
  security_groups = [aws_security_group.efs.id]
}

resource "helm_release" "aws_efs_csi_driver" {
  name       = "aws-efs-csi-driver"
  repository = "https://kubernetes-sigs.github.io/aws-efs-csi-driver"
  chart      = "aws-efs-csi-driver"
  namespace  = "kube-system"

  set {
    name  = "controller.serviceAccount.annotations.eks\\.amazonaws\\.com/role-arn"
    value = module.efs_csi_irsa_role.iam_role_arn
  }
}

# StorageClass for EFS model cache
resource "kubernetes_storage_class" "efs_model_cache" {
  metadata {
    name = "efs-model-cache"
  }
  storage_provisioner = "efs.csi.aws.com"
  parameters = {
    provisioningMode = "efs-ap"
    fileSystemId     = aws_efs_file_system.model_cache.id
    directoryPerms   = "755"
  }
  reclaim_policy      = "Retain"    # Don't delete model cache on PVC deletion
  volume_binding_mode = "Immediate"
}

# ─────────────────────────────────────────────
# GPU Operator via Helm
# ─────────────────────────────────────────────

resource "helm_release" "gpu_operator" {
  name             = "gpu-operator"
  repository       = "https://helm.ngc.nvidia.com/nvidia"
  chart            = "gpu-operator"
  version          = var.gpu_operator_version
  namespace        = "gpu-operator"
  create_namespace = true

  values = [file("${path.module}/../../../../configs/kubernetes/gpu-operator/values.yaml")]

  depends_on = [module.eks, helm_release.karpenter]
}

# ─────────────────────────────────────────────
# VPC
# ─────────────────────────────────────────────

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${var.cluster_name}-vpc"
  cidr = var.vpc_cidr

  azs             = data.aws_availability_zones.available.names
  private_subnets = var.private_subnet_cidrs
  public_subnets  = var.public_subnet_cidrs

  enable_nat_gateway   = true
  single_nat_gateway   = false    # HA: one NAT per AZ
  enable_dns_hostnames = true

  public_subnet_tags = {
    "kubernetes.io/role/elb"                    = 1
    "karpenter.sh/discovery"                    = var.cluster_name
  }
  private_subnet_tags = {
    "kubernetes.io/role/internal-elb"           = 1
    "karpenter.sh/discovery"                    = var.cluster_name
  }

  tags = local.common_tags
}

# ─────────────────────────────────────────────
# Supporting resources
# ─────────────────────────────────────────────

data "aws_availability_zones" "available" {
  state = "available"
}

module "ebs_csi_irsa_role" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name             = "${var.cluster_name}-ebs-csi"
  attach_ebs_csi_policy = true

  oidc_providers = {
    ex = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:ebs-csi-controller-sa"]
    }
  }
}

module "efs_csi_irsa_role" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name             = "${var.cluster_name}-efs-csi"
  attach_efs_csi_policy = true

  oidc_providers = {
    ex = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:efs-csi-controller-sa"]
    }
  }
}

resource "aws_security_group" "efs" {
  name        = "${var.cluster_name}-efs"
  description = "EFS model cache access from EKS nodes"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 2049
    to_port         = 2049
    protocol        = "tcp"
    security_groups = [module.eks.node_security_group_id]
  }

  tags = local.common_tags
}

locals {
  common_tags = {
    Cluster     = var.cluster_name
    ManagedBy   = "terraform"
    Environment = var.environment
    Project     = "ai-inference"
  }
}
