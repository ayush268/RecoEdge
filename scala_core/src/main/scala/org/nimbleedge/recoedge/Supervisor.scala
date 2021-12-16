package org.nimbleedge.recoedge

import models._

import akka.actor.typed.ActorRef
import akka.actor.typed.Behavior
import akka.actor.typed.scaladsl.Behaviors
import akka.actor.typed.scaladsl.ActorContext
import akka.actor.typed.scaladsl.AbstractBehavior
import akka.actor.typed.Signal
import akka.actor.typed.PostStop

object Supervisor {
    def apply(): Behavior[Command] =
        Behaviors.setup[Command](new Supervisor(_))

    sealed trait Command

    // Creating + Getting the actor references
    final case class RequestOrchestrator(requestId: Long, orcId: OrchestratorIdentifier, replyTo: ActorRef[OrchestratorRegistered])
        extends Supervisor.Command
    
    final case class OrchestratorRegistered(requestId: Long, actor: ActorRef[Orchestrator.Command])

    final case class RequestAggregator(requestId: Long, aggId: AggregatorIdentifier, replyTo: ActorRef[AggregatorRegistered])
        extends Supervisor.Command
        with Orchestrator.Command
    
    final case class AggregatorRegistered(requestId: Long, actor: ActorRef[Aggregator.Command])

    final case class RequestTrainer(requestId: Long, traId: TrainerIdentifier, replyTo: ActorRef[TrainerRegistered])
        extends Supervisor.Command
        with Orchestrator.Command
        with Aggregator.Command
    
    final case class TrainerRegistered(requestId: Long, actor: ActorRef[Trainer.Command])
    
    // In case of an Orchestrator Termination
    private final case class OrchestratorTerminated(actor: ActorRef[Orchestrator.Command], orcId: OrchestratorIdentifier)
        extends Supervisor.Command

    // Requesting Topology
    final case class RequestTopology(requestId: Long, entity: Identifier, replyTo: ActorRef[Topology])
        extends Supervisor.Command
        with Orchestrator.Command
        with Aggregator.Command
    
    // TODO
    final case class Topology(requestId: Long)

    // Start cycle
    // TODO
    final case class StartCycle(requestId: Long, replyTo: ActorRef[RespondModel]) extends Supervisor.Command
    final case class RespondModel(requestId: Long)

    // TODO
    // Add more messages
}

class Supervisor(context: ActorContext[Supervisor.Command]) extends AbstractBehavior[Supervisor.Command](context) {
    import Supervisor._

    // TODO
    // Topology
    // State Information

    context.log.info("Supervisor Started")

    override def onMessage(msg: Command): Behavior[Command] =
        msg match {
            case trackMsg @ RequestOrchestrator(requestId, orcId, replyTo) =>
                // TODO
                this
            
            case trackMsg @ RequestAggregator(requestId, aggId, replyTo) =>
                // TODO
                this
            
            case trackMsg @ RequestTrainer(requestId, traId, replyTo) =>
                // TODO
                this
            
            case trackMsg @ RequestTopology(requestId, entity, replyTo) =>
                // TODO
                this
            
            case StartCycle(requestId, replyTo) =>
                // TODO
                this
            
            case OrchestratorTerminated(actor, orcId) =>
                context.log.info("Orchestrator with id {} has been terminated", orcId.toString())
                // TODO
                this
        }

    override def onSignal: PartialFunction[Signal,Behavior[Command]] = {
        case PostStop =>
            context.log.info("Supervisor Stopped")
            this
    }
}