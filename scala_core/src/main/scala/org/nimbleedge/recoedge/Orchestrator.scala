package org.nimbleedge.recoedge

import models._

import akka.actor.typed.ActorRef
import akka.actor.typed.Behavior
import akka.actor.typed.scaladsl.Behaviors
import akka.actor.typed.scaladsl.ActorContext
import akka.actor.typed.scaladsl.AbstractBehavior
import akka.actor.typed.Signal
import akka.actor.typed.PostStop

object Orchestrator {
  def apply(orcId: OrchestratorIdentifier): Behavior[Command] =
    Behaviors.setup(new Orchestrator(_, orcId))

  trait Command

  // In case any Aggregator Termination
  private final case class AggregatorTerminated(actor: ActorRef[Aggregator.Command], aggId: AggregatorIdentifier)
    extends Orchestrator.Command

  // TODO
  // Add messages here
}

class Orchestrator(context: ActorContext[Orchestrator.Command], orcId: OrchestratorIdentifier) extends AbstractBehavior[Orchestrator.Command](context) {
  import Orchestrator._
  import Supervisor.{ RequestAggregator, RequestTrainer, RequestTopology }

  // TODO
  // Add state and persistent information

  context.log.info("Orchestrator {} started", orcId.toString())

  override def onMessage(msg: Command): Behavior[Command] =
    msg match {
      case trackMsg @ RequestAggregator(requestId, aggId, replyTo) =>
        // TODO
        this

      case trackMsg @ RequestTrainer(requestId, traId, replyTo) =>
        // TODO
        this
      
      case trackMsg @ RequestTopology(requestId, entity, replyTo) =>
        // TODO
        this
      
      case AggregatorTerminated(actor, aggId) =>
        context.log.info("Aggregator with id {} has been terminated", aggId.toString())
        // TODO
        this
    }
  
  override def onSignal: PartialFunction[Signal,Behavior[Command]] = {
    case PostStop =>
      context.log.info("Orchestrator {} stopeed", orcId.toString())
      this
  }
}