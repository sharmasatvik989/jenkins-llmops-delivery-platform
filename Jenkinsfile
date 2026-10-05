pipeline {
  agent any

  environment {
    IMAGE = "${DOCKER_REGISTRY}/jenkins-llmops-delivery-platform:${BUILD_NUMBER}"
  }

  stages {
    stage('Test') {
      steps {
        sh 'python -m pip install -r requirements.txt'
        sh 'python -m pytest -q'
      }
    }
    stage('Build image') {
      steps {
        sh 'docker build --pull --tag $IMAGE .'
      }
    }
    stage('Validate release') {
      steps {
        sh 'helm lint deploy/helm/inference-service'
        sh 'helm template inference deploy/helm/inference-service --set image.repository=$DOCKER_REGISTRY/jenkins-llmops-delivery-platform --set image.tag=$BUILD_NUMBER > /tmp/release.yaml'
      }
    }
    stage('Publish') {
      when { branch 'main' }
      steps {
        sh 'docker push $IMAGE'
      }
    }
    stage('Deploy') {
      when { branch 'main' }
      steps {
        sh 'helm upgrade --install inference deploy/helm/inference-service --namespace llmops --create-namespace --set image.repository=$DOCKER_REGISTRY/jenkins-llmops-delivery-platform --set image.tag=$BUILD_NUMBER --atomic --timeout 5m'
      }
    }
  }

  post {
    always { archiveArtifacts artifacts: '/tmp/release.yaml', allowEmptyArchive: true }
  }
}

